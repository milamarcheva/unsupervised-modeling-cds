package induction;

import java.util.*;
import fig.basic.*;

public class Options {
  public enum ModelType { gmm, pmmm, hmm, pcfg, dmv, seg, align, event3 };
  public enum InitType { random, bait, supervised, uniformz, artificial, hard, soft };
  public enum InputFormat { raw, tag, mrg, seg };
  public enum AlignmentModel { m1, m2, hmm };

  // PartIIIBegin
  public enum InductionType{ normal, morph , mlu, bg};
  // PartIIIEnd

  // Input
  @Option public ArrayList<String> inputPaths = new ArrayList();
  @Option public ArrayList<String> inputLists = new ArrayList();
  @Option public ArrayList<String> testInputPaths = new ArrayList();
  @Option public ArrayList<String> testInputLists = new ArrayList();
  @Option public String inputFileExt;
  @Option public String inputFileExt2; // For word alignment, the extension of the other language
  @Option(gloss="Description file for initializing artificial parameters") public String artificialDescriptionPath;
  @Option(gloss="Format of input", required=true) public InputFormat inputFormat;
  @Option(gloss="Maximum number of examples") public int maxExamples = Integer.MAX_VALUE;
  @Option(gloss="Maximum number of test examples") public int testMaxExamples = Integer.MAX_VALUE;
  @Option(gloss="Maximum length of an example") public int maxExampleLength = Integer.MAX_VALUE;
  @Option(gloss="For parsing") public boolean useTagsAsWords = false;
  @Option(gloss="Segment characters (default is segmenting words)") public boolean segChars = false;
  @Option(gloss="For segmentation") public String segBoundary = "||";
  @Option(gloss="Penalize words") public double segPenalty = 0.5;
  @Option(gloss="Maximum number of examples to use for extracting phrases (to limit # phrases)") public int maxExamplesForPhrases = Integer.MAX_VALUE;

  // Model
  @Option(gloss="Model", required=true) public ModelType modelType;
  @Option(gloss="Number of hidden states") public int K = 5;
  @Option(gloss="Maximum phrase length") public int maxPhraseLength = 5;
  @Option public AlignmentModel alignmentModel = AlignmentModel.m1;
  @Option(gloss="Threshold for posterior decoding") public double posteriorThreshold = 0.5;
  @Option public double gmmVariance = 1;
  @Option public double gmmGenRange = 10;
  @Option(gloss="PCFGs: tree structure is fixed") public boolean fixBracketing = false;
  @Option(gloss="PCFGs: fix the preterminal (POS) tags") public boolean fixPreTags = false;
  @Option(gloss="PCFGs: collapse the nonterminal tag set") public boolean collapseNonTags = false;
  @Option(gloss="PCFGs: number of hidden preterminal states") public int pK = 5;
  @Option(gloss="PCFGs: add CCM potentials (generate content and context)") public boolean useCCMPotentials = false;
  @Option(gloss="PCFGs: have the CCM generation depend on the state") public boolean ccmDependsOnState = false;
  @Option public boolean removePunctuation = false;
  @Option public boolean alignAgreement = true;

  //// Event3 model
  @Option(gloss="Output for external evaluation script") public boolean fullPredForEval = false;
  @Option(gloss="Random matching - baseline") public boolean fullPredRandomBaseline = false;
  @Option public Random fullPredRandom = new Random(1);

  @Option public String[] excludedFields = new String[0]; // List of <event type name>.<field name>
  @Option public String[] excludedEventTypes = new String[0]; // List of <event type name>
  @Option(gloss="Take the first event of each type (don't do this)") public boolean takeOneOfEventType = false;
  @Option public boolean treatCatAsSym = false;
  @Option public boolean useOnlyLabeledExamples = false;

  // Changes to the model can happen during training, so each of these specifies a starting and ending iteration for the
  // corresponding flag to be on
  @Option public Pair<Integer,Integer> indepEventTypes = new Pair(0, 0);
  @Option public Pair<Integer,Integer> indepFields = new Pair(0, 0);
  @Option(gloss="Each word chooses event type/field independently (word alignment, no segmentation)")
    public Pair<Integer,Integer> newEventTypeFieldPerWord = new Pair(0, 0);
  @Option(gloss="Each word chooses field independently (no segmentation at the field level)")
    public Pair<Integer,Integer> newFieldPerWord = new Pair(0, 0);
  @Option public Pair<Integer,Integer> oneEventPerExample = new Pair(0, 0);
  @Option public Pair<Integer,Integer> oneFieldPerEvent = new Pair(0, 0);

  @Option(gloss="p(t | w)") public boolean includeEventTypeGivenWord = false;
  @Option(gloss="Allow a field to show up twice in a row") public boolean disallowConsecutiveRepeatFields = false;

  // At event-type level
  @Option(gloss="p(include event? | e, v)") public boolean useEventSalienceModel = false;
  @Option(gloss="p(t) or p(t | t0)") public boolean useEventTypeDistrib = true;

  // At field level
  @Option(gloss="Generate and constrain the set of fields which are used for each event mentioned on these event types")
    public String[] useFieldSetsOnEventTypes = new String[0];
  @Option public Pair<Integer,Integer> useFieldSets = new Pair(0, 0);
  @Option public int minFieldSetSize = 0;
  @Option public int maxFieldSetSize = Integer.MAX_VALUE;

  // Tracks
  @Option(gloss="Tracks (for NFL): each track gets a subset of the event types")
    public String[] eventTypeTracks = new String[] { "ALL" }; // Default is one track with everything
  @Option(gloss="Jointly decide whether to have non-none event") public boolean jointEventTypeDecision = false;

  // Modify
  @Option public boolean includeEventTypeAsSymbol = false;
  @Option(gloss="Stem symbols, strings, and words") public boolean stemAll = false;
  @Option(gloss="Annotate and generate numeric quantities with labels (the word that follows)") public Pair<Integer,Integer> genLabels = new Pair(0, 0);

  @Option public String wordRolesPath;
  @Option(gloss="List of <event type name>.<field name>") public String[] useWordRolesOnFields = new String[0];

  // Specific control ofsmoothing
  @Option public double noneEventTypeSmoothing = 0;
  @Option public double fixedNoneEventTypeProb = Double.NaN;
  @Option public double fixedGenericProb = Double.NaN;
  @Option public double noneFieldSmoothing = 0;
  @Option public double fieldNameSmoothing = 0;
  @Option public boolean discountCatEmissions = false;

  @Option public boolean andIsPunctuation = false;
  @Option(gloss="A entity segment only break on punctuation") public boolean onlyBreakOnPunctuation = false;
  @Option(gloss="A entity segment cannot cross punctuation") public boolean dontCrossPunctuation = false;
  @Option(gloss="Limit field length (e.g. num and sym must be 1)") public boolean limitFieldLength = false;
  @Option(gloss="For each line, make a separate example (NFL data)") public boolean oneExamplePerLine = false;

  @Option public boolean debug = false;

  // Generic
  @Option public int trainStart = 0;
  @Option public int trainEnd = Integer.MAX_VALUE;
  @Option public int testStart = 0;
  @Option public int testEnd = 0;

  // Learning
  @Option public InitType initType = InitType.random;
  @Option public double initSmoothing = 0.01;
  @Option public double initNoise = 1e-3;
  @Option public Random initRandom = new Random(1);
  @Option(gloss="Randomness for permuting data points in online") public Random onlinePermRandom = new Random(1);
  @Option public boolean onlinePerm = false;
  @Option(gloss="If batch size is larger this threshold, store counts rather than infer states") public int batchSizeNewCounts = 1000;

  @OptionSet(name="stage1") public LearnOptions stage1 = new LearnOptions();
  @OptionSet(name="stage2") public LearnOptions stage2 = new LearnOptions();
  @Option(gloss="Output every this number of iterations") public int outputIterFreq = 1;
  @Option(gloss="Output full predictions (for debugging)") public boolean outputFullPred = false;
  @Option(gloss="Output training objective and test predictions on current parameters") public boolean outputCurrentState = false;
  @Option(gloss="Output every this number of examples (for online)") public double outputExampleFreq = 1000;
  @Option public int outputNumParamsPerVec = Integer.MAX_VALUE;
  @Option(gloss="This computation involves lots of logs and can be slow") public boolean computeELogZEntropy = false;

  // Generate artificial data
  @Option public int genNumExamples = 0;
  @Option(gloss="Maximum number of tokens per example") public int genMaxTokens = 100;
  @Option public Random genRandom = new Random(1);
  @Option public Random genInitRandom = new Random(1);
  @Option public InitType genInitType = InitType.supervised;

  @Option public int artNumWords = 100;
  @Option public double artAlpha = 0.5;

  // General
  @Option(gloss="Number of parameters per state to print") public int numOutputParams = 10;
  @Option(gloss="Number of threads to use") public int numThreads = 4;

  @Option(gloss="Mila") public int mila = 4;
  // PartIIIBegin
  @Option(gloss="Induction type: normal | morph | mlu ") public InductionType inductionType = InductionType.normal;
  @Option(gloss="Dilute value for shared cluster") public Integer diluteValue=2;
  @Option(gloss="Normalize, default is optimize, i.e. normalization + smoothing") public Boolean normalize = Boolean.FALSE;
//  @Option(gloss="Set of morphemes whose probs need to be diluted in the stage function") public ArrayList<String> diluteSet= new ArrayList();
  @Option(gloss="Order in which the morphemes are staged, 0: Brown") public Integer order = 0;
  @Option(gloss="Order in which the morphemes are staged as string, default is Brown") public String orderStr = "ING,IN,ON,PLU,IRPAST,POS,UNCCOP,ART,RPAST,R3,IR3,UNCAUX,CCOP,CAUX";
  @Option(gloss="Priming (anchoring) with morphemes, default is true") public Boolean priming =  Boolean.TRUE;
  @Option(gloss="Gradual unlocking, default is true") public Boolean gradualUnlocking =  Boolean.TRUE;

  @Option(gloss="Grouping of clusters, ensure right amount of K, 0: part III groupings") public Integer grouping = 0;
  @Option(gloss="Number of examples to use for staging, max 4295") public Integer stagingExamplesNum = 0;
  @Option(gloss="Fraction of examples to use for staging") public Double stagingExamplesFraction = 0.0;
  @Option(gloss="Only noun and verb initialization, no stages") public Boolean onlyNounVerb = Boolean.FALSE;

  // PartIII anchors
  @Option(gloss="Number of nouns for initialisation") public Integer nounsNum = 75;
  @Option(gloss="Number of verbs for initialisation") public Integer verbsNum = 53;

  @Option(gloss="Use one morpheme only as anchor") public Boolean anchor1 = Boolean.FALSE;

  @Option(gloss="Anchor index for uncontractible copular, 0-4") public Integer UNCCOPAnchorIndex = 0;
  @Option(gloss="Anchor index for uncontractible auxiliary, 0-4") public Integer UNCAUXAnchorIndex = 0;
  @Option(gloss="Anchor index for uncontractible auxiliary, 0-2") public Integer CAUXAnchorIndex = 0;
  @Option(gloss="Anchor index for uncontractible copular, 0-2") public Integer CCOPAnchorIndex = 0;
  @Option(gloss="Anchor index for preposition, 0-1") public Integer PREPAnchorIndex = 0;
  @Option(gloss="Anchor index for article, 0-1") public Integer ARTAnchorIndex = 0;
  @Option(gloss="Anchor index for irregualr past, 0-12") public Integer IRPASTAnchorIndex = 0;
  @Option(gloss="Anchor index for irregualr third person singular, 0-2") public Integer IR3AnchorIndex = 0;


  // PartIIIEnd

}
