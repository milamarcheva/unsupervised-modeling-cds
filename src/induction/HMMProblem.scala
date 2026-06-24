package induction

import java.util.Random
import fig.basic.{IOUtils, Indexer, StatFig}
import fig.exec.Execution
import fig.exec.Execution.putLogRec
import fig.record.Record
import fig.basic.LogInfo.{dbg, end_track, error, fail, logs, track}
import tea._
import _root_.tea.Utils.{assertValid, begin_track, fmt, fmt1, foreach, map, returnFirst}
import InductionUtils._
import scala.collection.mutable.ListBuffer

case class HMMProblem(opts: Options) extends TaggingProblem {
  type Widget = Array[Int]
  //  override def W = wordIndexer.size // Number of words

  case class Params(starts: ProbVec, transitions: Array[ProbVec], emissions: Array[ProbVec]) extends AParams {
    def foreachVec(f: (ProbVec => Any)) = { //applying the function f to every element of the probability vector
      f(starts) //starts is 1D
      transitions.foreach(f(_)) //transitions and emissions are 2D, hence the additional foreach
      emissions.foreach(f(_))
    }

    def output(puts: (String => Any)) = {
      Utils.foreachSorted(starts.getProbs, true, { (a: Int, v: Double) =>
        puts(String.format("S %s\t%s", fmt(a), fmt(v)))
      })
      puts("")

      foreach(K, { a: Int =>
        Utils.foreachSorted(transitions(a).getProbs, true, { (b: Int, v: Double) =>
          puts(String.format("T %s %s\t%s", fmt(a), fmt(b), fmt(v)))
        })
        puts("")
      })

      foreach(K, { a: Int =>
        Utils.foreachSorted(emissions(a).getProbs, opts.numOutputParams, true, { (w: Int, v: Double) =>
          puts(String.format("E %s %s\t%s", fmt(a), wstr(w), fmt(v)))
        })
        puts("")
      })
    }


  }


  class Performance extends TaggingPerformance[Widget] {
    def add(trueTags: Widget, predStates: Widget): Unit =
      if (trueTags != null)
        foreach(trueTags, predStates, { (t: Int, k: Int) => incr(t, k) })
  }

  type Example = WordExample[Widget]

  class InferState(override val ex: Example, override val params: Params, override val counts: Params, override val ispec: InferSpec) extends
    AHypergraphInferState[Widget, Example, Params](ex, params, counts, ispec) {
    def words = ex.words

    def N = words.length

    def newWidget = new Widget(N)

    def createHypergraph(H: Hypergraph[Widget]) = {
      def allowed(i: Int, a: Int) = !trueInfer || ex.trueWidget(i) == a

      def gen(i: Int, a: Int): Object = { // Generate the rest of the sequence from position i with state a
        if (i == N - 1)
          H.endNode
        else {
          val node = (i, a)
          if (H.addSumNode(node)) {
            foreach(K, { b: Int =>
              if (allowed(i + 1, b))
                H.addEdge(node, gen(i + 1, b), new Info {
                  def getWeight = get(params.transitions(a), b) * get(params.emissions(b), words(i + 1))

                  def setPosterior(v: Double) = {
                    update(counts.transitions(a), b, v)
                    update(counts.emissions(b), words(i + 1), v)
                  }

                  def choose(widget: Widget) = {
                    widget(i + 1) = b;
                    widget
                  }
                })
            })
          }
          node
        }
      }

      foreach(K, { a: Int =>
        if (allowed(0, a))
          H.addEdge(H.sumStartNode, gen(0, a), new Info { // Generate the state at position 0
            def getWeight = get(params.starts, a) * get(params.emissions(a), words(0))

            def setPosterior(v: Double) = {
              update(counts.starts, a, v)
              update(counts.emissions(a), words(0), v)
            }

            def choose(widget: Widget) = {
              widget(0) = a;
              widget
            }
          })
      })
    }
  }

  object InferState {
    def unapply[Widget, Example, Params](x: InferState): Some[InferState] =
      Some(x)
  }

  class Model extends TaggingModel[Widget, Params, Performance, Example, InferState] {
    def widgetToIntSeq(widget: Widget) = widget

    override def tokensToExample(tokens: Array[String], add: (Example => Any)) = {
      opts.inputFormat match {
        case Options.InputFormat.raw => //UNSUPERVISED case, cannot generate examples
          // word word word ...
          if (tokens.length <= opts.maxExampleLength)
            add(new Example(getIndex(wordIndexer, tokens), null))
        case Options.InputFormat.tag =>
          // word tag word tag word ...
          if (tokens.length / 2 <= opts.maxExampleLength)
            add(new Example(getIndex(wordIndexer, Utils.slice(tokens, 0, tokens.size, 2)),
              getIndex(tagIndexer, Utils.slice(tokens, 1, tokens.size, 2))))
      }
    }

    def exampleToString(ex: Example) =
      Utils.createArray({ add: (String => Any) =>
        foreach(ex.words.length, { i: Int =>
          add(wstr(ex.words(i))) //
          add(tstr(ex.trueWidget(i)))
        })
      }).mkString(" ")

    override def readExamples(path: String, maxExamples: Int, add: (Example => Any)) = {
      if (opts.inputFormat == Options.InputFormat.mrg)
        foreachTree(opts, path, maxExamples, { tree: Tree =>
          val words = tree.getYield
          if (words.size <= opts.maxExampleLength)
            add(new Example(getIndex(wordIndexer, Utils.toArray(words)),
              getIndex(tagIndexer, Utils.toArray(tree.getPreTerminalYield))))
        })
      else
        super.readExamples(path, maxExamples, add)
    }


    //Part III extension
    override def selectExamples(start: Int, end: Int): Unit = {
      val oldExamples = examples
      if (start < 0 || end < start || end > oldExamples.length)
        fail("Invalid example range [" + start + ", " + end + ") for " + oldExamples.length + " loaded examples")
      examples = oldExamples.slice(start, end)
    }

    override def lockStates(first: Int, last: Int): Unit = {
      track("Lock cluster from first (inclusive) to K")

      var i = first
      while (i < last) {
        params.starts.addCountKeepNonNegative_!(i, -100000000)
        foreach(i, { j: Int => params.transitions(j).addCountKeepNonNegative_!(i, -100000000) })
        foreach(last, { j: Int => params.transitions(i).addCountKeepNonNegative_!(j, -100000000) })
        foreach(W, { w: Int => params.emissions(i).addCountKeepNonNegative_!(0, -100000000) })

        i += 1
      }

      var fname = "initLOCKED" + first.toString + ".params"
      params.output(Execution.getFile(fname))

      //params.output(Execution.getFile("initLOCKED.params"))
      end_track
    }

    override def unlockState(state:Int): Unit = {
      track("Unlock state " + state)

      foreach(W, { w: Int => params.emissions(state).addCountKeepNonNegative_!(w,1) })

      var fname = "initUNLOCKED" + state.toString + ".params"
      params.output(Execution.getFile(fname))

      end_track
    }

    override def allEmissionsForWord(puts: (String => Any)) = { //words: Array[String],
      var words = Array("underwear", "measles", "airplane", "guitar", "racket", "movie", "sword", "sofa", "ant",
        "motorcycle", "hotel", "shirt", "apartment", "museum", "necklace", "spoon", "tea", "pear", "pepper", "peach",
        "pen", "cat", "saw", "flower", "newspaper", "fork", "dog", "bird", "grape", "necktie", "butterfly", "radish",
        "tent", "coffee", "chair", "net", "boot", "cup", "skate", "paper", "plate", "truck", "cow", "bean", "drum",
        "magazine", "onion", "tree", "bus", "horse", "weight", "knife", "novel", "potato", "lamp", "drill", "sock",
        "milk", "piano", "ring", "gun", "scarf", "pencil", "apple", "bed", "fish", "ball", "bomb", "mouse", "shoe",
        "hammer", "table", "grain", "bread", "watch", "drink", "eat", "bite", "suck", "spit", "vomit", "blow",
        "breathe", "laugh", "see", "hear", "know", "think", "smell", "fear", "sleep", "live", "die", "kill", "fight",
        "hunt", "hit", "cut", "split", "scratch", "dig", "swim", "fly", "walk", "come", "lie", "sit", "stand", "turn",
        "fall", "give", "hold", "squeeze", "rub", "wash", "wipe", "pull", "push", "throw", "tie", "sew", "count",
        "say", "sing", "play", "float", "freeze", "swell", "am", "are", "is", "was", "were", "in",
        "on", "ing", "ed", "ate", "drank", "did", "knew", "bit", "came", "fell", "froze", "gave", "sang", "swam",
        "threw","a", "the", "has", "does", "goes", "s" , "'re","'s", "'m" )

      var wis = ListBuffer[Int]()

      val emissionLength = params.emissions(0).getProbs.length
      for (word <- words) {
        var wi_n = wordIndexer.indexOf(word)
        if (wi_n >= 0 && wi_n < emissionLength) { // check if word is in the current emission vector
          wis += wi_n;
        }
      }

      for (wi <- wis) {
        var k = 0
        while (k < K) {
          puts(String.format("E %s %s\t%s", fmt(k), wstr(wi), fmt(params.emissions(k).getProbs(wi))))
          k += 1
        }
        puts("")
      }

    }

    override def stageGeneral(name: String, words: Array[String], cluster: Int, wordIndexerLength: Int,
                              countsType: String, dilute: Int, normalize: Boolean,
                              initStage: Boolean, anchor1: Boolean, UNCCOPAnchorIndex: Int,
                              UNCAUXAnchorIndex: Int, CAUXAnchorIndex: Int, CCOPAnchorIndex: Int,
                              PREPAnchorIndex: Int, ARTAnchorIndex: Int, IRPASTAnchorIndex: Int,
                              IR3AnchorIndex: Int, BGPLUAnchorIndex: Int, BGDEFAnchorIndex: Int,
                              BGPRESAnchorIndex: Int, BGINFAnchorIndex: Int, BGPASTAnchorIndex: Int,
                              BGPREPAnchorIndex: Int): Unit = {

      track("stage" + name + ": ")
      val emissionLength = params.emissions(0).getProbs.length
      var wi_length = if (wordIndexerLength > 0) wordIndexerLength else emissionLength
      var wis = ListBuffer[Int]()

      for (n <- words) {
        var wi_n = wordIndexer.getIndex(n)
        if (wi_n < wi_length) {
          wis += wi_n;
        }
      }

      var count = 0

      if (anchor1 && words.length > 1) {
        val anchorIndex: Int = name match {
          case "UNCCOP" => UNCCOPAnchorIndex
          case "UNCAUX" => UNCAUXAnchorIndex
          case "CAUX" => CAUXAnchorIndex
          case "CCOP" => CCOPAnchorIndex
          case "PREP" => PREPAnchorIndex
          case "ART" => ARTAnchorIndex
          case "IRPAST" => IRPASTAnchorIndex
          case "IR3" => IR3AnchorIndex
          case "BGPLU" => BGPLUAnchorIndex
          case "BGDEF" => BGDEFAnchorIndex
          case "BGPRES" => BGPRESAnchorIndex
          case "BGINF" => BGINFAnchorIndex
          case "BGPAST" => BGPASTAnchorIndex
          case "BGPREP" => BGPREPAnchorIndex
          // case _ => throw _root_.tea.Utils.fail("Unsupported anchor1 stage: " + name)
        }

        val wi = wordIndexer.indexOf(words(anchorIndex))

        if (countsType == "entireCluster") {
          count = wi_length / 1
        }
        else if (countsType == "sharedCluster") { //cluster is shared across several stages
          count = wi_length / (dilute)
        }
        track(name + " count: " + count + ", total words in wordIndexer: " + W + ", the anchor is " + words(anchorIndex) + " cluster number: " + cluster)

        if (wi >= 0 && wi < emissionLength)
          params.emissions(cluster).addCount_!(wi, count)
        else
          logs("Skipping missing anchor word for %s: %s", name, words(anchorIndex))
      }
      else {
        if (countsType == "entireCluster") { //cluster is only associated with the morphemes of the current stage
          count = wi_length / words.length
        }
        else if (countsType == "sharedCluster") { //cluster is shared across several stages
          count = wi_length / (dilute * words.length)
        }
        track(name + " count: " + count + ", total words in wordIndexer: " + W + ", total vocab count: " + words.length + " cluster number: " + cluster)

        for (wi <- wis) {
          params.emissions(cluster).addCount_!(wi, count)
        }
      }

      if (normalize) {
        params.normalize_!
      }
      else {
        params.optimize_!(opts.initSmoothing)
      }

      if (initStage) {
        params.output(Execution.getFile(name + ".params"))
      }
      else {

        params.output(Execution.getFile("stage" + name + ".params"))
      }
    }


    //PartIIIEnd

    def newInferState(ex: Example, params: Params, counts: Params, ispec: InferSpec) = new InferState(ex, params, counts, ispec)

    def newPerformance = new Performance

    def newParams = new Params(ProbVec.zeros(K), ProbVec.zeros2(K, K), ProbVec.zeros2(K, W)) //start, transition, emission

    def genExample = {
      val N = opts.genMaxTokens
      val states = new Array[Int](N)
      states(0) = genSample(params.starts)
      foreach(1, N, { i: Int => states(i) = genSample(params.transitions(states(i - 1))) })
      val words = map(N, { i: Int => genSample(params.emissions(states(i))) })
      new Example(words, states)
    }

    override def genExamples: Unit = {
      super.genExamples
    }

    override def learn(name: String, lopts: induction.LearnOptions): Unit = {
      super.learn(name, lopts)
    }

    override def readExamples: Unit = {
      super.readExamples
    }

  }

  def newModel = new Model

}
