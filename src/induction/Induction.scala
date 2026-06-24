package induction

import scala.collection.mutable.HashMap
import java.util.Random

import fig.basic.{IOUtils, Indexer, StatFig}
import fig.exec.Execution
import fig.exec.Execution.putLogRec
import fig.record.Record
import fig.basic.LogInfo.{logs, dbg, fail, error, track, end_track}

import tea._
import tea.Utils.{map, foreach, fmt, fmt1, returnFirst, assertValid}

import InductionUtils._

import util.control.Breaks._

class InductionRunner extends Runnable {
  val opts = new Options

  val nounsAll = Array("underwear", "measles", "airplane", "guitar", "racket", "movie", "sword", "sofa", "ant", "motorcycle", "hotel", "shirt", "apartment", "museum", "necklace", "spoon", "tea", "pear", "pepper", "peach", "pen", "cat", "saw", "flower", "newspaper", "fork", "dog", "bird", "grape", "necktie", "butterfly", "radish", "tent", "coffee", "chair", "net", "boot", "cup", "skate", "paper", "plate", "truck", "cow", "bean", "drum", "magazine", "onion", "tree", "bus", "horse", "weight", "knife", "novel", "potato", "lamp", "drill", "sock", "milk", "piano", "ring", "gun", "scarf", "pencil", "apple", "bed", "fish", "ball", "bomb", "mouse", "shoe", "hammer", "table", "grain", "bread", "watch")
  val verbsAll = Array("drink", "eat", "bite", "suck", "spit", "vomit", "blow", "breathe", "laugh", "see", "hear", "know", "think", "smell", "fear", "sleep", "live", "die", "kill", "fight", "hunt", "hit", "cut", "split", "scratch", "dig", "swim", "fly", "walk", "come", "lie", "sit", "stand", "turn", "fall", "give", "hold", "squeeze", "rub", "wash", "wipe", "pull", "push", "throw", "tie", "sew", "count", "say", "sing", "play", "float", "freeze", "swell")

  val nounsAllBG = Array("баба", "мама", "дядо", "работа", "глава", "приятел", "път", "човек", "дете", "чичо", "татко", "бебе", "куче", "мамо", "леля", "ръка", "глас", "майка", "вода", "дума", "цвете", "вечер", "нощ")
  val verbsAllBG = Array("помисля", "кажа", "има", "нямам", "видя-(се)", "река", "знам", "извикам", "дойда", "попитам", "запитам", "мога", "стана", "чакам", "казвам-(се)", "имам", "ставам", "чуя", "гледам", "искам", "разбирам-(се)", "погледна", "въздъхна")

  val orders = Map(0 -> List("ING", "IN", "ON", "PLU", "IRPAST", "POS", "UNCCOP", "ART", "RPAST", "R3", "IR3", "UNCAUX", "CCOP", "CAUX"),
    5555 -> List("ING", "PREP", "PLU", "IRPAST", "POS", "UNCCOP", "ART", "RPAST", "R3", "IR3", "UNCAUX", "CCOP", "CAUX"), 
    359 -> List("BGPLU", "BGDEF", "BGPRES", "BGINF", "BGPAST", "BGFUT", "BGPREP")
  )

  def run = {
    val prob = opts.modelType match {
      case Options.ModelType.gmm => new GMMProblem(opts)
      case Options.ModelType.pmmm => new ProdMultMixtureProblem(opts)
      case Options.ModelType.hmm => new HMMProblem(opts)
      case Options.ModelType.pcfg => new PCFGProblem(opts)
      case Options.ModelType.dmv => new DMVProblem(opts)
      case Options.ModelType.seg => new SegmentationProblem(opts)
      case Options.ModelType.align => new WordAlignmentProblem(opts)
      case Options.ModelType.event3 => new Event3Problem(opts)
      case _ => fail("Unknown model type: " + opts.modelType); null
    }
    val model = prob.newModel
    model.readExamples
    if (model.numExamples == 0 && opts.genNumExamples == 0)
      fail("No examples were read. Check -inputPaths/-inputLists and run from the unsupervised-modeling directory.")
//    model.selectExamples(0, 102828)

    Record.begin("stats")
    track("Stats", true)
    model.logStats
    end_track
    Record.end

    if (opts.genNumExamples > 0) { // Use generated examples instead
      model.preInit
      model.init(opts.genInitType, opts.genInitRandom)
      model.genExamples
    }
    model.preInit
    model.init(opts.initType, opts.initRandom)

    // PartIIIBegin

    //    with On
    var starts = Map("UNCCOP" -> 178252, "CAUX" -> 286988, "RPAST" -> 242183, "PREP" -> 122481, "IN" -> 122481, "UNCAUX" -> 256854, "ING" -> 102828, "IRPAST" -> 153573, "POS" -> 173957, "PLU" -> 136905, "ART" -> 196114, "R3" -> 247283, "CCOP" -> 263783, "ON" -> 130292, "IR3" -> 251822)
    var ends = Map("UNCCOP" -> 196114, "CAUX" -> 295245, "RPAST" -> 247283, "IN" -> 130292, "UNCAUX" -> 263783, "ING" -> 122481, "IRPAST" -> 173957, "POS" -> 178252, "PLU" -> 153573, "ART" -> 242183, "R3" -> 251822, "CCOP" -> 286988, "PREP" -> 136905, "ON" -> 136905, "IR3" -> 256854)
    // 295245
    // Derived from sample-data/bg/bg_hmm_tokenised_morphLemma_duplicate_category_ranges.tsv.
    // startsBG are inclusive start indices from the TSV, and endsBG are exclusive
    // end indices to match selectExamples(start, end) -> slice(start, end).
    var startsBG = Map("BGPLU" -> 29420, "BGDEF" -> 52669, "BGPRES" -> 113838, "BGINF" -> 154055, "BGPAST" -> 173394, "BGFUT" -> 203801, "BGPREP" -> 213467)
    var endsBG = Map("BGPLU" -> 52669, "BGDEF" -> 113838, "BGPRES" -> 154055, "BGINF" -> 173394, "BGPAST" -> 203801, "BGFUT" -> 213467, "BGPREP" -> 239244)
    // testStart = 239244

    if (opts.stagingExamplesNum>0){
      starts.keys.foreach ((m) => ends+= (m -> (starts.apply(m)+ opts.stagingExamplesNum)))
      track("UNCCOP should be 196114 + num; num is " + opts.stagingExamplesNum + " and ends(UNCCOP) is " + ends.apply("UNCCOP") )
    } else if (opts.stagingExamplesFraction>0){
      starts.keys.foreach((m) => ends += (m -> (starts.apply(m)+((ends.apply(m) - starts.apply(m))*opts.stagingExamplesFraction).toInt)))
      track("UNCCOP has 6929 examples; frac is " + opts.stagingExamplesFraction + " and ends(UNCCOP) is " + ends.apply("UNCCOP"))
    }

    //I had removed on from order 0

    val clusterGroupings = Map(
      0 -> Set(Set("ING"), Set("IN", "ON"), Set("PLU"), Set("IRPAST"), Set("POS", "ART"), Set("UNCCOP", "CCOP"), Set("UNCAUX", "CAUX"), Set("RPAST", "R3", "IR3")), //10
      1 -> Set(Set("ING"), Set("IN", "ON"), Set("PLU"), Set("IRPAST"), Set("POS"), Set("ART"), Set("UNCCOP", "CCOP"), Set("UNCAUX", "CAUX"), Set("RPAST", "R3", "IR3")), //11 clusters!!
      2 -> Set(Set("ING"), Set("IN", "ON"), Set("PLU", "ART"), Set("IRPAST"), Set("POS"), Set("UNCCOP", "CCOP"), Set("UNCAUX", "CAUX"), Set("RPAST", "R3", "IR3")), //10
      3 -> Set(Set("ING"), Set("IN", "ON"), Set("PLU"), Set("POS", "ART"), Set("UNCCOP", "CCOP"), Set("UNCAUX", "CAUX"), Set("IRPAST", "RPAST", "R3", "IR3")), //10
      4 -> Set(Set("ING"), Set("IN", "ON"), Set("PLU"), Set("POS", "ART"), Set("UNCCOP", "CCOP", "UNCAUX", "CAUX"), Set("IRPAST", "RPAST", "R3", "IR3")), //9!!
      5 -> Set(Set("ING"), Set("PREP"), Set("PLU"), Set("IRPAST"), Set("POS", "ART"), Set("UNCCOP", "CCOP"), Set("UNCAUX", "CAUX"), Set("RPAST", "R3", "IR3")), //10, PREP instead of IN on
      359 -> Set(Set("BGPLU"), Set("BGDEF"), Set("BGPRES", "BGPAST", "BGFUT"), Set("BGINF"), Set("BGPREP")) // 8 total 
    )

    val clusterIndices = Map(
      0 -> Map("UNCCOP" -> -1, "CAUX" -> -1, "RPAST" -> -1, "IN" -> -1, "UNCAUX" -> -1, "ING" -> -1, "IRPAST" -> 1, "POS" -> -1, "PLU" -> -1, "ART" -> -1, "R3" -> -1, "CCOP" -> -1, "ON" -> -1, "IR3" -> -1),
      1 -> Map("UNCCOP" -> -1, "CAUX" -> -1, "RPAST" -> -1, "IN" -> -1, "UNCAUX" -> -1, "ING" -> -1, "IRPAST" -> 1, "POS" -> -1, "PLU" -> -1, "ART" -> -1, "R3" -> -1, "CCOP" -> -1, "ON" -> -1, "IR3" -> -1),
      2 -> Map("UNCCOP" -> -1, "CAUX" -> -1, "RPAST" -> -1, "IN" -> -1, "UNCAUX" -> -1, "ING" -> -1, "IRPAST" -> 1, "POS" -> -1, "PLU" -> -1, "ART" -> -1, "R3" -> -1, "CCOP" -> -1, "ON" -> -1, "IR3" -> -1),
      3 -> Map("UNCCOP" -> -1, "CAUX" -> -1, "RPAST" -> -1, "IN" -> -1, "UNCAUX" -> -1, "ING" -> -1, "IRPAST" -> -1, "POS" -> -1, "PLU" -> -1, "ART" -> -1, "R3" -> -1, "CCOP" -> -1, "ON" -> -1, "IR3" -> -1),
      4 -> Map("UNCCOP" -> -1, "CAUX" -> -1, "RPAST" -> -1, "IN" -> -1, "UNCAUX" -> -1, "ING" -> -1, "IRPAST" -> -1, "POS" -> -1, "PLU" -> -1, "ART" -> -1, "R3" -> -1, "CCOP" -> -1, "ON" -> -1, "IR3" -> -1),
      5 -> Map("UNCCOP" -> -1, "CAUX" -> -1, "RPAST" -> -1, "PREP" -> -1, "UNCAUX" -> -1, "ING" -> -1, "IRPAST" -> 1, "POS" -> -1, "PLU" -> -1, "ART" -> -1, "R3" -> -1, "CCOP" -> -1 , "IR3" -> -1), 
      359 -> Map("BGPLU" -> -1, "BGDEF" -> -1, "BGPRES" -> -1, "BGINF" -> -1, "BGPAST" -> -1, "BGFUT" -> -1, "BGPREP" -> -1)
    )
//
//    val orders = Map(0 -> List("ING", "IN", "ON", "PLU", "IRPAST", "POS", "UNCCOP", "ART", "RPAST", "R3", "IR3", "UNCAUX", "CCOP", "CAUX"),
//      5555 -> List("ING", "PREP", "PLU", "IRPAST", "POS", "UNCCOP", "ART", "RPAST", "R3", "IR3", "UNCAUX", "CCOP", "CAUX")
//    )
//    List(ING, IN, ON, PLU, IRPAST, POS, UNCCOP, ART, RPAST, R3, IR3, UNCAUX, CAUX, CCOP)
//    List(ING, IN, ON, PLU, IRPAST, POS, UNCCOP, ART, RPAST, R3, IR3, UNCAUX, CCOP, CAUX)
//    var order = orders.apply(opts.order)

    //UNncomment the line below if you want order from string
    // var order = opts.orderStr.split(",").toList


    val clusterGrouping = clusterGroupings.apply(opts.grouping)
    var clusterIndex = clusterIndices.apply(opts.grouping)

    var clusterIndexBG = clusterIndices.apply(359)
    val clusterGroupingBG = clusterGroupings.apply(359)

    val priming = opts.priming

    //20251103 commenting this out to test with random words priming
//    var clusterMorphemes = Map(
//      "UNCCOP" -> Array("am", "are", "is", "was", "were"),
//      "UNCAUX" -> Array("am", "are", "is", "was", "were"),
//      "CAUX" -> Array("'m", "'re", "'s"),
//      "CCOP" -> Array("'m", "'re", "'" +
//        "s"),
//      "PREP" -> Array("in", "on"),
//      "IN" -> Array("in"),
//      "ON" -> Array("on"),
//      "ING" -> Array("ing"),
//      "RPAST" -> Array("ed"),
//      "IRPAST" -> Array("saw", "ate", "drank", "did", "knew", "bit", "came", "fell", "froze", "gave", "sang", "swam", "threw"),
//      "POS" -> Array("'s"),
//      "PLU" -> Array("s"),
//      "ART" -> Array("a", "the"),
//      "R3" -> Array("s"),
//      "IR3" -> Array("has", "does", "goes")
//    )

    //these are random -- for priming wth random experiment
    var clusterMorphemes = Map(
      "UNCCOP" -> Array("0a"),
      "UNCAUX" -> Array("0is"),
      "CAUX" -> Array("aaah"),
      "CCOP" -> Array("aaargh"),
      "PREP" -> Array("aaaro"),
      "ING" -> Array("aaacho"),
      "RPAST" -> Array("aahahb"),
      "IRPAST" -> Array("aahhb"),
      "POS" -> Array("aaoohb"),
      "PLU" -> Array("ab"),
      "ART" -> Array("ababahb"),
      "R3" -> Array("ababahb"),
      "IR3" -> Array("abahbahbahbahb")
    )

    var clusterMorphemesBG = Map(
      "BGPLU" -> Array("ове",  "и", "та" ),
      "BGDEF" -> Array("ът", "ят", "та", "то", "те", "а", "я"),
      "BGPRES" -> Array("а", "еш", "е", "ем", "ете", "ат", "я", "иш", "и", "им", "ите", "ят", "м", "ш", "ме", "те", "т"),
      "BGINF" -> Array("л", "ла", "ло", "ли"),
      "BGPAST" -> Array("х", "ха", "хме", "хте", "ше"),
      "BGFUT" -> Array("ще"),
      "BGPREP" -> Array("на", "за")
    )


    val clustersDilute = Set("IN", "ON", "PREP", "IRPAST", "POS", "UNCCOP", "ART", "RPAST", "R3", "IR3", "UNCAUX", "CCOP", "CAUX")

    val englishInitEnd = 102828
    val englishTrainEnd = 295245

    val bulgarianInitEnd = 29420
    val bulgarianTrainEnd = 239244

    val usingBG = opts.useBG || opts.inductionType == Options.InductionType.bg

    val defaultOrder = if (usingBG) orders(359) else orders(opts.order)
    val explicitOrderStr = Option(opts.orderStr).map(_.trim).getOrElse("")

    var order =
      if (explicitOrderStr.nonEmpty)
        explicitOrderStr.split(",").map(_.trim).filter(_.nonEmpty).toList
      else
        defaultOrder

    val activeNouns = if (usingBG) nounsAllBG else nounsAll
    val activeVerbs = if (usingBG) verbsAllBG else verbsAll

    val activeStarts = if (usingBG) startsBG else starts
    val activeEnds = if (usingBG) endsBG else ends

    val activeInitEnd = if (usingBG) bulgarianInitEnd else englishInitEnd
    val activeTrainEnd = if (usingBG) bulgarianTrainEnd else englishTrainEnd

    val activeClusterGrouping = if (usingBG) clusterGroupingBG else clusterGrouping
    var activeClusterIndex = if (usingBG) clusterIndexBG else clusterIndex

    val activeClusterMorphemes = if (usingBG) clusterMorphemesBG else clusterMorphemes
    val activeClustersDilute = if (usingBG) activeClusterMorphemes.keySet else clustersDilute

    val invalidStages = order.filterNot(activeStarts.contains)
    if (invalidStages.nonEmpty) {
      val languageLabel = if (usingBG) "Bulgarian" else "English"
      fail("Order contains stages not available for " + languageLabel + ": " + invalidStages.mkString(", "))
    }

    if (opts.inductionType == Options.InductionType.normal) {
      model.learn("stage1", opts.stage1)
      model.learn("stage2", opts.stage2)
//      track("Writing out emissions for selected words for this stage", Execution.getFile("stage2.emissions"))
//      Utils.writeLines(Execution.getFile("stage2.emissions"), model.allEmissionsForWord)
      end_track
    }
    else if (priming!=true) { // gradual unlocking True + ordering True
      model.selectExamples(0, activeInitEnd)
      model.lockStates(3, prob.opts.K)
      model.learn("stage0", opts.stage2)
      var lastLocked = 3


        for (m <- order) {
          track(m + "lastLocked"+ (lastLocked))
          model.readExamples
          model.selectExamples(activeStarts.apply(m), activeEnds.apply(m))
          //unlock state
          if (lastLocked<10){
            model.unlockState(lastLocked)
          }

          var grouping = Set[String]()
          var newInd = 0

          if (activeClusterIndex.apply(m) > 0) {
            //lastLocked should not be updated because no more new clusters need to be unlocked
          } else {
            breakable {
              for (g <- activeClusterGrouping) {
                if (g.contains(m)) {
                  grouping = g
                  break
                }
              }
            }
            var groupingNoM = grouping - m
            breakable {
              for (m2 <- groupingNoM) { //grouping-m
                if (activeClusterIndex.apply(m2) > 0) {
                  newInd = activeClusterIndex.apply(m2)
                  break
                }
              }
            }
            if (newInd == 0) {
              for (m3 <- grouping) { //all morphems in the group
                activeClusterIndex += (m3 -> lastLocked)
              }
              lastLocked += 1
            } else {
              for (m3 <- grouping) { //all morphems in the group
                activeClusterIndex += (m3 -> newInd)
              }
            }
          }

          model.lockStates(lastLocked, prob.opts.K)
          model.learn("locked" + lastLocked, opts.stage2)

          end_track
        }
      model.readExamples
      model.selectExamples(opts.testStart, opts.testEnd)
      opts.stage2.numIters = 0
      model.learn("test", opts.stage2)


    }
    else if (opts.gradualUnlocking!=true) { // Gradual unlocking is false + priming is True
      val nouns = activeNouns.slice(0, opts.nounsNum)
      val verbs = activeVerbs.slice(0, opts.verbsNum)
      track("orderStr " + order)

      // original Brown order
      model.selectExamples(0, activeInitEnd)
      model.stageGeneral("initNOUN", nouns, 0, initStage = true)
      model.stageGeneral("initVERB", verbs, 1, initStage = true)



        model.learn("stage0", opts.stage2)
        var lastLocked = 3

        for (m <- order) {
          model.readExamples
          model.selectExamples(activeStarts.apply(m), activeEnds.apply(m))

          var grouping = Set[String]()
          var newInd = 0

          if (activeClusterIndex.apply(m) > 0) {
            //lastLocked should not be updated because no more new clusters need to be unlocked
          } else {
            breakable {
              for (g <- activeClusterGrouping) {
                if (g.contains(m)) {
                  grouping = g
                  break
                }
              }
            }
            var groupingNoM = grouping - m
            breakable {
              for (m2 <- groupingNoM) { //grouping-m
                if (activeClusterIndex.apply(m2) > 0) {
                  newInd = activeClusterIndex.apply(m2)
                  break
                }
              }
            }
            if (newInd == 0) {
              for (m3 <- grouping) { //all morphems in the group
                activeClusterIndex += (m3 -> lastLocked)
              }
              lastLocked += 1
            } else {
              for (m3 <- grouping) { //all morphems in the group
                activeClusterIndex += (m3 -> newInd)
              }
            }
          }

            model.stageGeneral(m, activeClusterMorphemes.apply(m), activeClusterIndex.apply(m), countsType = "sharedCluster",
              dilute = opts.diluteValue, normalize = opts.normalize, anchor1 = opts.anchor1,
              UNCCOPAnchorIndex = opts.UNCCOPAnchorIndex, UNCAUXAnchorIndex = opts.UNCAUXAnchorIndex,
              CAUXAnchorIndex = opts.CAUXAnchorIndex, CCOPAnchorIndex = opts.CCOPAnchorIndex,
              PREPAnchorIndex = opts.PREPAnchorIndex, ARTAnchorIndex = opts.ARTAnchorIndex,
              IRPASTAnchorIndex = opts.IRPASTAnchorIndex, IR3AnchorIndex = opts.IR3AnchorIndex,
              BGPLUAnchorIndex = opts.BGPLUAnchorIndex, BGDEFAnchorIndex = opts.BGDEFAnchorIndex,
              BGPRESAnchorIndex = opts.BGPRESAnchorIndex, BGINFAnchorIndex = opts.BGINFAnchorIndex,
              BGPASTAnchorIndex = opts.BGPASTAnchorIndex, BGPREPAnchorIndex = opts.BGPREPAnchorIndex)

          model.learn("stage" + m, opts.stage2)

          track("Writing out emissions for selected words for this stage", Execution.getFile("stage" + m + ".emissions"))
          Utils.writeLines(Execution.getFile("stage" + m + ".emissions"), model.allEmissionsForWord)


          end_track
        }
      model.readExamples
      model.selectExamples(opts.testStart, opts.testEnd)
      opts.stage2.numIters = 0
      model.learn("test", opts.stage2)


    }
    else if (opts.inductionType == Options.InductionType.morph) {
      val nouns = activeNouns.slice(0, opts.nounsNum)
      val verbs = activeVerbs.slice(0, opts.verbsNum)
      track("orderStr " + order  )

      // original Brown order
      model.selectExamples(0, activeInitEnd)
      model.stageGeneral("initNOUN", nouns, 0, initStage = true)
      model.stageGeneral("initVERB", verbs, 1, initStage = true)

      if (opts.onlyNounVerb){
        model.learn("stage0", opts.stage2)
        model.readExamples
        model.selectExamples(activeInitEnd, activeTrainEnd)
        model.learn("stageAllMorphemes" , opts.stage2)
        track("Writing out emissions for selected words for this stage", Execution.getFile("stageAllMorphemes" + ".emissions"))
        //Utils.writeLines(Execution.getFile("stageAllMorphemes" + ".emissions"), model.allEmissionsForWord)
        end_track
      }

      else {
        model.lockStates(3, prob.opts.K)
        model.learn("stage0", opts.stage2)
        var lastLocked = 3

        for (m <- order) {
          model.readExamples
//          if (m == "CAUX"){
//            track("this is CAUX")
//            model.selectExamples(starts.apply(m), ends.apply(m)+3)
//          }
//          else{
          model.selectExamples(activeStarts.apply(m), activeEnds.apply(m))

//          }
          var grouping = Set[String]()
          var newInd = 0

          if (activeClusterIndex.apply(m) > 0) {
            //lastLocked should not be updated because no more new clusters need to be unlocked
          } else {
            breakable {
              for (g <- activeClusterGrouping) {
                if (g.contains(m)) {
                  grouping = g
                  break
                }
              }
            }
            var groupingNoM = grouping - m
            breakable {
              for (m2 <- groupingNoM) { //grouping-m
                if (activeClusterIndex.apply(m2) > 0) {
                  newInd = activeClusterIndex.apply(m2)
                  break
                }
              }
            }
            if (newInd == 0) {
              for (m3 <- grouping) { //all morphems in the group
                activeClusterIndex += (m3 -> lastLocked)
              }
              lastLocked += 1
            } else {
              for (m3 <- grouping) { //all morphems in the group
                activeClusterIndex += (m3 -> newInd)
              }
            }
          }

          if (activeClustersDilute.contains(m)) {
            model.stageGeneral(m, activeClusterMorphemes.apply(m), activeClusterIndex.apply(m), countsType = "sharedCluster",
              dilute = opts.diluteValue, normalize = opts.normalize, anchor1 = opts.anchor1,
              UNCCOPAnchorIndex = opts.UNCCOPAnchorIndex, UNCAUXAnchorIndex = opts.UNCAUXAnchorIndex,
              CAUXAnchorIndex = opts.CAUXAnchorIndex, CCOPAnchorIndex = opts.CCOPAnchorIndex,
              PREPAnchorIndex = opts.PREPAnchorIndex, ARTAnchorIndex = opts.ARTAnchorIndex,
              IRPASTAnchorIndex = opts.IRPASTAnchorIndex, IR3AnchorIndex = opts.IR3AnchorIndex,
              BGPLUAnchorIndex = opts.BGPLUAnchorIndex, BGDEFAnchorIndex = opts.BGDEFAnchorIndex,
              BGPRESAnchorIndex = opts.BGPRESAnchorIndex, BGINFAnchorIndex = opts.BGINFAnchorIndex,
              BGPASTAnchorIndex = opts.BGPASTAnchorIndex, BGPREPAnchorIndex = opts.BGPREPAnchorIndex)
          }
          else {
            model.stageGeneral(m, activeClusterMorphemes.apply(m), activeClusterIndex.apply(m), normalize = opts.normalize, anchor1 = opts.anchor1,
              UNCCOPAnchorIndex = opts.UNCCOPAnchorIndex, UNCAUXAnchorIndex = opts.UNCAUXAnchorIndex,
              CAUXAnchorIndex = opts.CAUXAnchorIndex, CCOPAnchorIndex = opts.CCOPAnchorIndex,
              PREPAnchorIndex = opts.PREPAnchorIndex, ARTAnchorIndex = opts.ARTAnchorIndex,
              IRPASTAnchorIndex = opts.IRPASTAnchorIndex, IR3AnchorIndex = opts.IR3AnchorIndex,
              BGPLUAnchorIndex = opts.BGPLUAnchorIndex, BGDEFAnchorIndex = opts.BGDEFAnchorIndex,
              BGPRESAnchorIndex = opts.BGPRESAnchorIndex, BGINFAnchorIndex = opts.BGINFAnchorIndex,
              BGPASTAnchorIndex = opts.BGPASTAnchorIndex, BGPREPAnchorIndex = opts.BGPREPAnchorIndex)
          }

          model.lockStates(lastLocked, prob.opts.K)
          model.learn("stage" + m, opts.stage2)

//          track("Writing out emissions for selected words for this stage", Execution.getFile("stage" + m + ".emissions"))
          Utils.writeLines(Execution.getFile("stage" + m + ".emissions"), model.allEmissionsForWord)


          end_track
        }

        model.readExamples
        model.selectExamples(opts.testStart, opts.testEnd)
        opts.stage2.numIters = 0
        model.learn("test", opts.stage2)

      }

    }
    else if (opts.inductionType == Options.InductionType.bg) {
      val nouns = activeNouns.slice(0, opts.nounsNum)
      val verbs = activeVerbs.slice(0, opts.verbsNum)
      track("orderStr " + order)

      model.selectExamples(0, activeInitEnd)
      model.stageGeneral("initNOUN", nouns, 0, initStage = true)
      model.stageGeneral("initVERB", verbs, 1, initStage = true)

      if (opts.onlyNounVerb) {
        model.learn("stage0", opts.stage2)
        model.readExamples
        model.selectExamples(activeInitEnd, activeTrainEnd)
        model.learn("stageAllMorphemes", opts.stage2)
        track("Writing out emissions for selected words for this stage", Execution.getFile("stageAllMorphemes" + ".emissions"))
        Utils.writeLines(Execution.getFile("stageAllMorphemes" + ".emissions"), model.allEmissionsForWord)
        end_track
      }

      else { // NO GRADUAL UNLOCKING
        model.learn("stage0", opts.stage2)
        var lastLocked = 3

        for (m <- order) {
          model.readExamples
          model.selectExamples(activeStarts.apply(m), activeEnds.apply(m))
          var grouping = Set[String]()
          var newInd = 0

          if (activeClusterIndex.apply(m) > 0) {
            //lastLocked should not be updated because no more new clusters need to be unlocked
          } else {
            breakable {
              for (g <- activeClusterGrouping) {
                if (g.contains(m)) {
                  grouping = g
                  break
                }
              }
            }
            var groupingNoM = grouping - m
            breakable {
              for (m2 <- groupingNoM) { //grouping-m
                if (activeClusterIndex.apply(m2) > 0) {
                  newInd = activeClusterIndex.apply(m2)
                  break
                }
              }
            }
            if (newInd == 0) {
              for (m3 <- grouping) { //all morphems in the group
                activeClusterIndex += (m3 -> lastLocked)
              }
              lastLocked += 1
            } else {
              for (m3 <- grouping) { //all morphems in the group
                activeClusterIndex += (m3 -> newInd)
              }
            }
          }

          model.stageGeneral(m, activeClusterMorphemes.apply(m), activeClusterIndex.apply(m), countsType = "sharedCluster",
            dilute = opts.diluteValue, normalize = opts.normalize, anchor1 = opts.anchor1,
            UNCCOPAnchorIndex = opts.UNCCOPAnchorIndex, UNCAUXAnchorIndex = opts.UNCAUXAnchorIndex,
            CAUXAnchorIndex = opts.CAUXAnchorIndex, CCOPAnchorIndex = opts.CCOPAnchorIndex,
            PREPAnchorIndex = opts.PREPAnchorIndex, ARTAnchorIndex = opts.ARTAnchorIndex,
            IRPASTAnchorIndex = opts.IRPASTAnchorIndex, IR3AnchorIndex = opts.IR3AnchorIndex,
            BGPLUAnchorIndex = opts.BGPLUAnchorIndex, BGDEFAnchorIndex = opts.BGDEFAnchorIndex,
            BGPRESAnchorIndex = opts.BGPRESAnchorIndex, BGINFAnchorIndex = opts.BGINFAnchorIndex,
            BGPASTAnchorIndex = opts.BGPASTAnchorIndex, BGPREPAnchorIndex = opts.BGPREPAnchorIndex)

          model.learn("stage" + m, opts.stage2)

//          track("Writing out emissions for selected words for this stage", Execution.getFile("stage" + m + ".emissions"))
//          Utils.writeLines(Execution.getFile("stage" + m + ".emissions"), model.allEmissionsForWord)


          end_track
        }
        model.readExamples
        model.selectExamples(opts.testStart, opts.testEnd)
        opts.stage2.numIters = 0
        model.learn("test", opts.stage2)
      }

    }


//    else if (opts.inductionType == Options.InductionType.mlu) {
//
//      val nouns = nounsAll.slice(0, opts.nounsNum)
//      val verbs = verbsAll.slice(0, opts.verbsNum)
//
//      model.selectExamples(0, activeInitEnd)
//      model.stageGeneral("initNOUN", nouns, 0, initStage = true)
//      model.stageGeneral("initVERB", verbs, 1, initStage = true)
//      model.lockStates(3, prob.opts.K)
//      model.learn("stageI", opts.stage2)
//      track("Writing out emissions for selected words for this stage", Execution.getFile("stageI.emissions"))
//      Utils.writeLines(Execution.getFile("stageI.emissions"), model.allEmissionsForWord)
//      end_track
//
//      //      model.initNOUN
//      //      model.learn("stageN", opts.stage2)
//      //      model.lockStates(2, prob.opts.K)
//      //      model.initVERB
//      //      model.lockStates(3, prob.opts.K)
//      //      model.learn("stageI", opts.stage2)
//
//      // MLU Stage II
//      model.readExamples
//      model.selectExamples(102828, 153573)
//      model.stageGeneral("ING", clusterMorphemes.apply("ING"), 3, countsType = "entireCluster", dilute = opts.diluteValue)
//      model.stageGeneral("IN", clusterMorphemes.apply("IN"), 4, countsType = "sharedCluster", dilute = opts.diluteValue)
//      model.stageGeneral("ON", clusterMorphemes.apply("ON"), 4, countsType = "sharedCluster", dilute = opts.diluteValue)
//      model.stageGeneral("PLU", clusterMorphemes.apply("PLU"), 5, countsType = "entireCluster", dilute = opts.diluteValue)
//      model.lockStates(6, prob.opts.K)
//      model.learn("stageII", opts.stage2)
//      track("Writing out emissions for selected words for this stage", Execution.getFile("stageII.emissions"))
//      Utils.writeLines(Execution.getFile("stageII.emissions"), model.allEmissionsForWord)
//      end_track
//
//      // MLU Stage III
//      model.readExamples
//      model.selectExamples(153573, 196114)
//      model.stageGeneral("IRPAST", clusterMorphemes.apply("IRPAST"), 1, countsType = "sharedCluster", dilute = opts.diluteValue)
//      model.stageGeneral("POS", clusterMorphemes.apply("POS"), 6, countsType = "sharedCluster", dilute = opts.diluteValue)
//      model.stageGeneral("UNCCOP", clusterMorphemes.apply("UNCCOP"), 7, countsType = "sharedCluster", dilute = opts.diluteValue)
//      model.lockStates(8, prob.opts.K)
//      model.learn("stageIII", opts.stage2)
//      track("Writing out emissions for selected words for this stage", Execution.getFile("stageIII.emissions"))
//      Utils.writeLines(Execution.getFile("stageIII.emissions"), model.allEmissionsForWord)
//      end_track
//
//      // MLU Stage IV
//      model.readExamples
//      model.selectExamples(196114, 251822)
//      model.stageGeneral("ART", clusterMorphemes.apply("ART"), 6, countsType = "sharedCluster", dilute = opts.diluteValue)
//      model.stageGeneral("RPAST", clusterMorphemes.apply("RPAST"), 8, countsType = "sharedCluster", dilute = opts.diluteValue)
//      model.stageGeneral("R3", clusterMorphemes.apply("R3"), 8, countsType = "sharedCluster", dilute = opts.diluteValue)
//      model.lockStates(9, prob.opts.K)
//      model.learn("stageIV", opts.stage2)
//      track("Writing out emissions for selected words for this stage", Execution.getFile("stageIV.emissions"))
//      Utils.writeLines(Execution.getFile("stageIV.emissions"), model.allEmissionsForWord)
//      end_track
//
//      // MLU Stage V
//      model.readExamples
//      model.selectExamples(251822, 295245)
//      model.stageGeneral("IR3", clusterMorphemes.apply("IR3"), 8, countsType = "sharedCluster", dilute = opts.diluteValue)
//      model.stageGeneral("UNCAUX", clusterMorphemes.apply("UNCAUX"), 9, countsType = "sharedCluster", dilute = opts.diluteValue)
//      model.stageGeneral("CCOP", clusterMorphemes.apply("CCOP"), 7, countsType = "sharedCluster", dilute = opts.diluteValue)
//      model.stageGeneral("CAUX", clusterMorphemes.apply("CAUX"), 9, countsType = "sharedCluster", dilute = opts.diluteValue)
//      model.learn("stageV", opts.stage2)
//      track("Writing out emissions for selected words for this stage", Execution.getFile("stageV.emissions"))
//      Utils.writeLines(Execution.getFile("stageV.emissions"), model.allEmissionsForWord)
//      end_track
//
//      //test
////      model.readExamples
////      model.selectExamples(295245, 295248)
//
//
//    }

    // PartIIIEnd

  }
}

object Induction {
  def main(args: Array[String]) = {
    val x = new InductionRunner
    fig.exec.Execution.run(args, x, x.opts)
  }
}
