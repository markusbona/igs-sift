/**
 * Extract and display Scale Invariant Features after the method of David Lowe
 * \cite{Lowe04} in an image.
 * 
 * BibTeX:
 * <pre>
 * &#64;article{Lowe04,
 *   author    = {David G. Lowe},
 *   title     = {Distinctive Image Features from Scale-Invariant Keypoints},
 *   journal   = {International Journal of Computer Vision},
 *   year      = {2004},
 *   volume    = {60},
 *   number    = {2},
 *   pages     = {91--110},
 * }
 * </pre>
 */

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Polygon;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;
import java.util.Vector;

import javax.imageio.ImageIO;
import javax.swing.JFrame;

import mpi.cbg.fly.Feature;
import mpi.cbg.fly.Filter;
import mpi.cbg.fly.FloatArray2D;
import mpi.cbg.fly.FloatArray2DSIFT;

import org.neuroph.core.learning.SupervisedTrainingElement;
import org.neuroph.core.learning.TrainingSet;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.util.TransferFunctionType;

public class CbirWithSift extends JFrame {
	// helper variables for the repaint
	IgsImage cur_image;

	// the extracted visual words - model for the VisualWordHistogram
	List<VisualWord> bagofwords = new Vector<VisualWord>();

	// a model to classify a VisualWordHistogram into a ImageClass
	Object decisionModel;

	// how many visual words should by classified
	private static int K = 300;

	// the minimum count of members in a "visual-word" class
	private static int MIN_CLASS_SIZE = 5;

	// how many images should be read from the input folders
	private static int readImages = 10000;
	// private static int readImages = 10000;

	// number of SIFT iterations: more steps will produce more features
	// default = 4
	private static int steps = 5;

	// for testing: delay time for showing images in the GUI
	private static int wait = 0;
	private static int waitVerify = 500;

	/**
	 * �* The method doLearnDecisionModel sets those according to the output of
	 * the nn. �
	 */
	static List<String> classNames;

	/**
	 * �* �* REPLACE THIS METHOD �* �* �* Classifies a VisualWordHistogram into
	 * an Image Class based on the learned �* model �* �* @param histogram �* �
	 * � � � � �the given VisualWordHistogram �* @return the name of the
	 * Image-Class �
	 */
	public String doClassifyImageContent(int[] histogram) {

		MultiLayerPerceptron myMlPerceptron = (MultiLayerPerceptron) decisionModel;

		// convert to double[]
		double[] dhistogram = new double[histogram.length];
		for (int i = 0; i < histogram.length; i++) {
			dhistogram[i] = histogram[i];
		}

		myMlPerceptron.setInput(dhistogram);
		myMlPerceptron.calculate();
		double[] output = myMlPerceptron.getOutput();

		// find biggest
		double maxOutput = -1;
		int maxOutputPos = -1;
		for (int i = 0; i < output.length; i++) {
			if (maxOutput < output[i]) {
				maxOutput = output[i];
				maxOutputPos = i;
			}
		}

		System.out.println("NN maxOutput:" + maxOutput + " pos:" + maxOutputPos
				+ " class:" + classNames.get(maxOutputPos));
		for (int i = 0; i < output.length; i++)
			System.out.print(i + ":" + output[i] + " ");
		System.out.println();

		if (maxOutputPos >= 0 && maxOutput > -1) {
			return classNames.get(maxOutputPos);
		} else {
			return "unknown";
		}

	}

	private static int[] findMinMax(Collection<Vector<int[]>> data) {
		int max = Integer.MIN_VALUE;
		int min = Integer.MAX_VALUE;
		// find min/max
		for (Vector<int[]> v : data) {
			for (int[] array : v) {
				for (int i = 0; i < array.length; i++) {
					if (max < array[i])
						max = array[i];
					if (min > array[i])
						min = array[i];
				}
			}
		}

		return new int[] { min, max };

	}

	private static void setClassNames(Set<String> keySet) {
		classNames = new LinkedList<String>();
		for (String className : keySet) {
			classNames.add(className);
		}
	}

	/**
	 * 
	 * REPLACE THIS METHOD
	 * 
	 * 
	 * Learns a model based on the training data set
	 * 
	 * @param dataSet
	 *            a list of VisualWordHistograms for each Image-Class
	 * @return a model for the Classifier
	 */
	public static Object doLearnDecisionModel(Map<String, Vector<int[]>> dataSet) {
		System.out.println("Learning Decision Model...");
		List<int[]> histoCollection = new LinkedList<int[]>();

		// Tuning parameter!
		double minSupport = 0.2;
		int numFeatures = 4;

		// Calculate Support of Visual words over all class
		for (String className : dataSet.keySet()) {
			histoCollection.addAll(dataSet.get(className));
		}

		System.out.println("Starting FIS with " + histoCollection.size()
				+ " Histograms");

		// Anzahl der histogramme
		HashMap<Histogram, Integer> histograms = new HashMap<Histogram, Integer>();
		for (int[] histo : histoCollection) {
			Histogram h = new Histogram(histo);
			if (!histograms.containsKey(h)) {
				histograms.put(h, 1);
			} else {
				histograms.put(h, histograms.get(h) + 1);
			}
		}
		System.out.println("Found " + histograms.size()
				+ " different Histograms");

		LinkedHashMap<Histogram, Double> frequencies = new LinkedHashMap<Histogram, Double>();
		LinkedHashMap<Histogram, Double> infrequencies = new LinkedHashMap<Histogram, Double>();
		LinkedHashMap<Integer, Double> frequentFeatures = new LinkedHashMap<Integer, Double>();
		LinkedHashMap<Integer, Double> infrequentFeatures = new LinkedHashMap<Integer, Double>();
		for (Histogram h : histograms.keySet()) {
			for (int f : h.features) {
				if (!frequentFeatures.containsKey(f) && 
					!infrequentFeatures.containsKey(f)) {
					Histogram hf = new Histogram(new int[] { f });
					double support = calculateSupport(histograms, hf);
					if (support >= minSupport) {
						frequentFeatures.put(f, support);
						frequencies.put(hf, support);
					} else {
						infrequentFeatures.put(f, support);
					}
				}
			}
		}

		System.out.println("Found " + frequentFeatures.size()
				+ " Frequent Features");

		for (int numberOfFeatures = 2; numberOfFeatures <= numFeatures; numberOfFeatures++) {
			System.out.println();
			System.out.println("Round " + (numberOfFeatures-1)+" with "+numberOfFeatures+" Features");

			// 4. build new entries from old entries, by adding single words
			System.out.println(frequencies.size()
					+ " entries. Combining new entries...");
			LinkedHashSet<Histogram> newFeatureSets = new LinkedHashSet<Histogram>(
					8192);
			for (Entry<Histogram, Double> e : frequencies.entrySet())
				for (int feature : frequentFeatures.keySet())
					// only add words that aren't already in the word set
					if (!e.getKey().contains(feature)) {
						List<Integer> features = new ArrayList<Integer>();
						for (int i = 0; i < e.getKey().features.length; i++)
							features.add(e.getKey().features[i]);
						features.add(feature);

						newFeatureSets.add(new Histogram(features));
					}

			// 5. remove sets with insufficient support
			System.out.println("Filtering " + newFeatureSets.size()
					+ " entries...");
			frequencies.clear();
			for (Histogram features : newFeatureSets) {
				if(!containsSubSet(infrequencies, features)) {
					double support = calculateSupport(histograms, features);
					if (support >= minSupport)
						frequencies.put(features, support);
					else {
						infrequencies.put(features, support);
					}
				}
			}
		}

		System.out.println();
		System.out.println("Found " + frequencies.size() + " entries");
		System.out.println();

		// NN (ohne frequent itemset)
		System.out.println("Initialising Neural Network ...");
		// set classNames
		setClassNames(dataSet.keySet());
		// normalize data minMax[0] == min value, minMax[1] == max value
		int[] minMax = findMinMax(dataSet.values());

		// create training set (logical XOR function)
		System.out.println("Build test data ...");
		TrainingSet<SupervisedTrainingElement> trainingSet = new TrainingSet<SupervisedTrainingElement>(
				K, classNames.size());

		int classNum = 0;
		for (Vector<int[]> v : dataSet.values()) {
			for (int[] data : v) {
				// convert to double[] and normalize
				double[] dData = new double[data.length];
				for (int i = 0; i < data.length; i++) {
					dData[i] = (data[i] - minMax[0])
							/ ((double) minMax[1] - minMax[0]);
					// value must be within 0 and 1
					if (dData[i] < 0 || dData[i] > 1) {
						System.err
								.println("nn data must be within 0 to 1, but is "
										+ dData[i]);
					}
				}
				// create Trainingset
				double[] output = new double[classNames.size()];
				for (int i = 0; i < output.length; i++)
					if (i == classNum)
						output[i] = 1;
					else
						output[i] = 0;
				trainingSet.addElement(new SupervisedTrainingElement(dData,
						output));
			}

			classNum++;
		}

		// create multi layer perceptron
		System.out.println("learn...");
		MultiLayerPerceptron nnet = new MultiLayerPerceptron(
				TransferFunctionType.TANH, K, K / 10, classNames.size());
		nnet.learn(trainingSet);

		System.out.println("learning done");
		return nnet;
	}

	private static boolean containsSubSet(HashMap<Histogram, Double> all,
			Histogram h) {
		for (Entry<Histogram, Double> subSet : all.entrySet()) {
			if (h.containsAll(subSet.getKey())) {
				return true;
			}
		}

		return false;
	}
	
	private static double calculateSupport(HashMap<Histogram, Integer> all,
			Histogram h) {
		double support = 0.0;

		double f = 1.0 / all.size();

		for (Entry<Histogram, Integer> query : all.entrySet()) {
			if (query.getKey().containsAll(h)) {
				support += f;
//				support += query.getValue() * f;
			}
		}

		return support;
	}

	/**
	 * 
	 * IMPLEMENT THIS METHOD
	 * 
	 * 
	 * Classifies a SIFT-Feature Vector into a VisualWord Class by finding the
	 * nearest visual word in the bagofwords "space"
	 * 
	 * @param f
	 *            a SIFT feature
	 * @return the class ID (0..k) or null if quality is not good enough
	 */
	public Integer doClassifyVisualWord(Feature f) {
		if (bagofwords == null || f == null)
			return null;

		/*
		 * Find best cluster
		 */
		// Index of best cluster
		Integer bestmatch = null;
		// Best cluster
		VisualWord bestWord = null;
		// Distance to best cluster so far
		float shortestDistance = Float.MAX_VALUE;

		for (int i = 0; i < bagofwords.size(); i++) {
			VisualWord word = bagofwords.get(i);
			float distance = word.centroied.descriptorDistance(f);

			if (bestmatch == null || distance < shortestDistance) {
				bestmatch = i;
				shortestDistance = distance;
				bestWord = word;
			}
		}

		/*
		 * Check distance quality (this has to be done for the best cluster
		 * only, thus it is not within the loop)
		 */
		// If a cluster has been found
		float alpha = 1.1f;
		if (bestWord != null) {
			// the @{link VisualWord.verficationValue} is the current average
			// distance
			// from cluster members to its centroid. The quality criteria is
			// that the distance
			// can at most be alpha * VisualWord.verficationValue, thus not only
			// slightly increasing
			// the clusters cohesion

			if (alpha * shortestDistance > (Float) bestWord.verificationValue) {
				bestmatch = null;
			}
		}

		return bestmatch;
	}

	/**
	 * 
	 * 
	 * IMPLEMENT THIS METHOD
	 * 
	 * 
	 * a k-mean clustering implementation for SIFT-Features: (float[] :
	 * Feature.descriptor)
	 * 
	 * @param _points
	 *            a list of all found features in the training set
	 * @param K
	 *            how many classes (visual words)
	 * @param minCount
	 *            the minimum number of members in each class
	 * @return the centroides of the k-mean = visual words list
	 */
	public static List<VisualWord> doClusteringVisualWords(
			final Feature[] _points, int K, int minCount) {
		System.out.println("Start clustering with: " + _points.length
				+ " pkt to " + K + " classes");

		List<VisualWord> centeroids = new LinkedList<VisualWord>();

		for (int i = 0; i < K; i++) {
			// take a random feature as start point
			Random rnd = new Random();
			int pos = rnd.nextInt(_points.length);
			VisualWord tmp = new VisualWord();
			tmp.centroied = _points[pos];
			centeroids.add(tmp);
		}

		//
		int newRandPos = 1;
		boolean testAfterCenter = false;
		while (newRandPos > 0) {
			for (int c = 0; c < K; c++)
				centeroids.get(c).nearFeature = new LinkedList<Feature>();

			// Allocate each point to the nearest cluster center
			for (int i = 0; i < _points.length; i++) {
				double distance = 0.0;
				int centerNr = 0;

				for (int j = 0; j < K; j++) {
					double d = _points[i]
							.descriptorDistance(centeroids.get(j).centroied);

					if (j == 0 || d < distance) {
						distance = d;
						centerNr = j;
					}
				}
				centeroids.get(centerNr).nearFeature.add(_points[i]);
			}

			//
			int nrOfDescElement = centeroids.get(0).centroied.descriptor.length;
			newRandPos = 0;
			for (int c = 0; c < K; c++) {
				int nrOfFeatures = centeroids.get(c).nearFeature.size();
				if (nrOfFeatures < minCount) {
					newRandPos++;
					// set to a new random position
					Random rnd = new Random();
					int pos = rnd.nextInt(_points.length);
					centeroids.get(c).centroied = _points[pos];
				}
			}

			// System.out.println(newRandPos);

			// move centeroid into the center
			if (newRandPos == 0) {
				float percent = 0.75f;

				for (int c = 0; c < K; c++) {
					int nrOfFeatures = centeroids.get(c).nearFeature.size();
					float[] distanceLimit = new float[nrOfFeatures];

					for (int i = 0; i < nrOfFeatures; i++) {
						distanceLimit[i] = centeroids.get(c).nearFeature
								.get(i)
								.descriptorDistance(centeroids.get(c).centroied);
					}
					Arrays.sort(distanceLimit);
					int pos = (int) Math.ceil(distanceLimit.length * percent);

					for (int d = 0; d < nrOfDescElement; d++) {
						double dx = 0.0;
						for (int f = 0; f < pos; f++) {
							dx += centeroids.get(c).nearFeature.get(f).descriptor[d];
						}
						centeroids.get(c).centroied.descriptor[d] = (float) (dx / pos);
					}
					centeroids.get(c).verificationValue = distanceLimit[pos];
				}

				if (!testAfterCenter) {
					testAfterCenter = true;
					newRandPos++;
				}
			}
		}

		return centeroids;
	}

	/* Do not change anything from here */

	// initial sigma
	private static float initial_sigma = 1.6f;
	// feature descriptor size
	private static int fdsize = 4;
	// feature descriptor orientation bins
	private static int fdbins = 8;
	// size restrictions for scale octaves, use octaves < max_size and >
	// min_size only
	private static int min_size = 64;
	private static int max_size = 1024;

	public CbirWithSift() throws IOException {
		super("Clustering");
		setVisible(true);
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		setSize(600, 600);

		Thread t = new Thread(new Runnable() {
			public void run() {
				try {

					setTitle("Learning: readData");
					LinkedList<IgsImage> trainingImages = readImages(
							"training", readImages);

					setTitle("Learning: VisualWord by Clustering");

					Vector<Feature> allLearnFeatchers = new Vector<Feature>();
					for (IgsImage i : trainingImages)
						allLearnFeatchers.addAll(i.features);

					long startTimeVW = System.currentTimeMillis();
					// calculate the visual words with k-means
					bagofwords = doClusteringVisualWords(
							(Feature[]) allLearnFeatchers
									.toArray(new Feature[0]), K, MIN_CLASS_SIZE);
					long endTimeVW = System.currentTimeMillis();

					setTitle("Show: visualWords in TraningsData");
					Map<String, Vector<int[]>> imageContentTrainingData = new HashMap<String, Vector<int[]>>();

					// create the VisiualWordHistograms for each training image
					for (IgsImage i : trainingImages) {
						if (!imageContentTrainingData.containsKey(i.className))
							imageContentTrainingData.put(i.className,
									new Vector<int[]>());
						int[] ImageVisualWordHistogram = new int[K];

						for (Feature f : i.features) {
							Integer wordClass = doClassifyVisualWord(f);
							if (wordClass != null)
								ImageVisualWordHistogram[wordClass.intValue()]++;
						}

						imageContentTrainingData.get(i.className).add(
								ImageVisualWordHistogram);

						cur_image = i;
						repaint();
						Thread.sleep(wait);
					}

					long startTimeDM = System.currentTimeMillis();
					setTitle("Learning: decisionModel");
					decisionModel = doLearnDecisionModel(imageContentTrainingData);
					long endTimeDM = System.currentTimeMillis();

					setTitle("Testing: readData");
					LinkedList<IgsImage> testImages = readImages("test",
							readImages);

					long startTime = System.currentTimeMillis();

					int success = 0;
					setTitle("Verify: test data");

					// create the VisiualWordHistograms for each test image and
					// classify it
					for (IgsImage i : testImages) {
						int[] ImageVisualWordHistogram = new int[K];

						for (Feature f : i.features) {
							Integer wordClass = doClassifyVisualWord(f);
							if (wordClass != null)
								ImageVisualWordHistogram[wordClass.intValue()]++;
						}

						i.classifiedName = doClassifyImageContent(
								ImageVisualWordHistogram).toString();

						if (i.isClassificationCorect())
							success++;

						cur_image = i;
						repaint();
						Thread.sleep(waitVerify);
					}

					long endTime = System.currentTimeMillis();

					System.out.println("Verified "
							+ (success / (double) testImages.size()) * 100
							+ "% in " + (endTime - startTime) + "ms");
					System.out.println("Learned " + K + " Visual Words in: "
							+ (endTimeVW - startTimeVW) + "ms!");
					System.out.println("Learned the image classification in: "
							+ (endTimeDM - startTimeDM) + "ms");

				} catch (Exception _e) {
					_e.printStackTrace();
				}
			}
		});
		t.setDaemon(true);
		t.start();
	}

	/**
	 * Reads maxImages from a folder, calculates the SIFT features and wraps the
	 * results into a IgsImage also paints each image on the GUI
	 * 
	 * @param folder
	 * @param maxImages
	 * @return the list of read IgsImages
	 * @throws IOException
	 * @throws InterruptedException
	 */
	LinkedList<IgsImage> readImages(String folder, int maxImages)
			throws IOException, InterruptedException {
		LinkedList<IgsImage> images = new LinkedList<IgsImage>();

		File actual = new File("./images/" + folder);

		int i = 0;

		for (File f : actual.listFiles()) {
			if (i++ > maxImages)
				break;
			IgsImage image = new IgsImage();
			image.image = ImageIO.read(f);
			image.className = f.getName()
					.substring(0, f.getName().indexOf('_'));
			image.features = calculateSift(image.image);

			cur_image = image;
			repaint();
			Thread.sleep(wait);

			images.add(image);
		}

		return images;
	}

	/**
	 * draws a rotated square with center point center, having size and
	 * orientation
	 */
	static void drawSquare(Graphics _g, double[] o, double scale,
			double orient, Integer _class) {
		scale /= 2;

		double sin = Math.sin(orient);
		double cos = Math.cos(orient);

		int[] x = new int[6];
		int[] y = new int[6];

		x[0] = (int) (o[0] + (sin - cos) * scale);
		y[0] = (int) (o[1] - (sin + cos) * scale);

		x[1] = (int) o[0];
		y[1] = (int) o[1];

		x[2] = (int) (o[0] + (sin + cos) * scale);
		y[2] = (int) (o[1] + (sin - cos) * scale);
		x[3] = (int) (o[0] - (sin - cos) * scale);
		y[3] = (int) (o[1] + (sin + cos) * scale);
		x[4] = (int) (o[0] - (sin + cos) * scale);
		y[4] = (int) (o[1] - (sin - cos) * scale);
		x[5] = x[0];
		y[5] = y[0];

		// if(_class==null || _class.intValue()==92 || _class.intValue()==69 ||
		// _class.intValue()==91) {

		_g.setColor(Color.red);
		_g.drawPolygon(new Polygon(x, y, x.length));
		_g.setColor(Color.yellow);
		if (_class != null)
			_g.drawString(_class + "", x[0], y[0]);
		// }

	}

	@Override
	public synchronized void paint(Graphics _g) {

		_g.clearRect(0, 0, 1000, 1000);

		if (cur_image == null)
			return;

		_g.drawImage(cur_image.image, 0, 0, null);

		_g.setColor(cur_image.isClassificationCorect() ? Color.green
				: Color.red);

		_g.drawString(cur_image.className + " > " + cur_image.classifiedName,
				20, cur_image.image.getHeight() + 40);

		if (cur_image.features != null)
			for (Feature f : cur_image.features)
				drawSquare(_g, new double[] { f.location[0], f.location[1] },
						fdsize * 4.0 * (double) f.scale,
						(double) f.orientation, doClassifyVisualWord(f));

	}

	public static FloatArray2D ImageToFloatArray2D(BufferedImage image)
			throws IOException {
		FloatArray2D image_float = null;

		int count = 0;
		image_float = new FloatArray2D(image.getWidth(), image.getHeight());

		for (int y = 0; y < image.getHeight(); y++) {
			for (int x = 0; x < image.getWidth(); x++) {
				int rgbV = image.getRGB(x, y);
				int b = rgbV & 0xff;
				rgbV = rgbV >> 8;
				int g = rgbV & 0xff;
				rgbV = rgbV >> 8;
				int r = rgbV & 0xff;
				image_float.data[count++] = 0.3f * r + 0.6f * g + 0.1f * b;
			}
		}

		return image_float;
	}

	public static void main(String[] _args) throws Exception {
		new CbirWithSift();
	}

	private Vector<Feature> calculateSift(BufferedImage image)
			throws IOException {

		Vector<Feature> _features = new Vector<Feature>();

		FloatArray2DSIFT sift = new FloatArray2DSIFT(fdsize, fdbins);

		FloatArray2D fa = ImageToFloatArray2D(image);
		Filter.enhance(fa, 1.0f);

		fa = Filter.computeGaussianFastMirror(fa,
				(float) Math.sqrt(initial_sigma * initial_sigma - 0.25));

		long start_time = System.currentTimeMillis();
		System.out.print("processing SIFT ...");

		sift.init(fa, steps, initial_sigma, min_size, max_size);
		_features = sift.run(max_size);

		System.out.println(" took " + (System.currentTimeMillis() - start_time)
				+ "ms to find \t" + _features.size() + " features");

		return _features;
	}

}
