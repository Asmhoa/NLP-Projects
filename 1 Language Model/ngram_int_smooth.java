/* 
 * Amol Sharma
 * Counterpart to ngram_no_smooth that has linear interpolation smoothing
 * The weights for each of the n-grams can be changed on line 26
 * Compile: javac part2.java
 * Run: java part2
 */

import java.io.*;
import java.util.*;

public class ngram_int_smooth {
    
    public static HashMap<String, Integer> unigramMap, bigramMap,
        trigramMap, unigramMapWithStart, bigramMapWithStart, currentMap;

    public static Scanner trainScanner;
    public static Scanner devScanner;
    public static Scanner testScanner;

    public static int unigramTrainingTokens = 0;
    public static HashSet<String> keysToUNK;

    public static int historyLength;
    public static final double lambda1 = 0.2, lambda2 = 0.6, lambda3 = 0.2;
    
    public static void main(String[] args) {

        try {
            unigramMap = new HashMap<String, Integer>(30000);
            bigramMap = new HashMap<String, Integer>(30000);
            trigramMap = new HashMap<String, Integer>(30000);

            for (int i = 1; i < 4; i++) {
                historyLength = i;
                train();
            }
            // System.out.println(unigramMap.size());
            test();

        } catch (FileNotFoundException e) {
            System.out.println("File not Found");
            return;
        }

    }

    public static void train() throws FileNotFoundException {
        switch (historyLength) {
            case 1:
                currentMap = unigramMap;
                break;
            case 2:
                currentMap = bigramMap;
                break;
            case 3:
                currentMap = trigramMap;
                break;
            default:
                break;
        }
        makeScanners();
        readTrainingFile();
    }

    public static void test() throws FileNotFoundException {
        makeScanners();
        reportPerplexity(trainScanner, "Training");
        reportPerplexity(devScanner, "Dev");
        reportPerplexity(testScanner, "Test");
    }

    // Read the file and count the occurences of the tokens
    public static void readTrainingFile() throws FileNotFoundException {
        int numStops = 0;
        unigramMap.putIfAbsent("\n", 0); // Stops

        // Once we advance, the words will cycle 
        String lastLastWord = null, lastWord = "<START>", token = "<START>";

        // int i = 0; // This was for halfing data
        while (trainScanner.hasNextLine()) { // && i < 30765) { // For each line
            // i++;
            String line = trainScanner.nextLine();
            Scanner lineScanner = new Scanner(line);

            do {
                // Advance
                lastLastWord = lastWord;
                lastWord = token;
                token = lineScanner.next();
                insertCounts(lastLastWord, lastWord, token);

                // Keep track of original training file without starts
                // Add counts for starts and stops later
                if (historyLength == 1) {
                    unigramTrainingTokens++;
                }

            } while (lineScanner.hasNext());

            // Advance once more per line for stop symbol
            lastLastWord = lastWord;
            lastWord = token;
            token = "\n";
            insertCounts(lastLastWord, lastWord, token);
            if (historyLength == 1) numStops++;
        }
        
        if (historyLength == 1) {
            storeUNKsFromUnigram();
            unigramTrainingTokens += numStops; // Add stops to count of tokens

            unigramMapWithStart = new HashMap<String, Integer>(unigramMap); // Copy
            unigramMapWithStart.put("<START>", numStops); // Add 1 start for each line
        } else if (historyLength == 2) {
            bigramMapWithStart = new HashMap<String, Integer>(bigramMap);
            bigramMapWithStart.put("<START> <START>", numStops); // Add 2 starts per line
        }
    }

    // Insert counts into maps
    public static void insertCounts(String lastLastWord, String lastWord,
        String token) {

        switch (historyLength) {
            case 1: // Unigram
                unigramMap.putIfAbsent(token, 0);
                unigramMap.put(token, unigramMap.get(token) + 1);
                break;
            case 2: // Bigram
                if (keysToUNK.contains(lastWord)) {lastWord = "UNK";}
                if (keysToUNK.contains(token)) {token = "UNK";}
                // Add current word combination
                String combination = lastWord + " " + token;
                bigramMap.putIfAbsent(combination, 0);
                bigramMap.put(combination, bigramMap.get(combination) + 1);   
                break;
            case 3: // Trigram
                if (keysToUNK.contains(lastLastWord)) {lastLastWord = "UNK";}
                if (keysToUNK.contains(lastWord)) {lastWord = "UNK";}
                if (keysToUNK.contains(token)) {token = "UNK";}
                // Add current word combination
                String comb = lastLastWord + " " + lastWord + " " + token;
                trigramMap.putIfAbsent(comb, 0);
                trigramMap.put(comb, trigramMap.get(comb) + 1);
                break;
            default:
                break;
        }
    }

    public static double get_pX(String lastLastWord, String lastWord,
        String token) {
        double px = 0.0; // If we havent seen it probability is still 0

        // Make copies of immutable ints if we need to redistribute weights
        double l1 = lambda1, l2 = lambda2, l3 = lambda3; 

        // UNK
        if (unigramMap.get(lastLastWord) == null) {lastLastWord = "UNK";} 
        if (unigramMap.get(lastWord) == null) {lastWord = "UNK";} 
        if (unigramMap.get(token) == null) {token = "UNK";}

        if (bigramMapWithStart.get(lastLastWord + " " + lastWord) ==  null) {
            // If we are here, we haven't seen the history for this trigram,
            // Still calculating by having a weight to l3 will give us a 
            // third term divided by 0. Since we can only have a bigram or trigram,
            // redistribute the weight so they still add to one in this case
            l1 += l3 / 2;
            l1 += l3 / 2;
            l3 = 0;
        }

        // Get Unigram theta
        px += l1 * ((double)unigramMap.get(token) / unigramTrainingTokens);

        // Get Bigram theta
        if (bigramMap.get(lastWord + " " + token) != null) {
            double numerator = (double) bigramMap.get(lastWord + " " + token);
            px += l2 * (numerator / unigramMapWithStart.get(lastWord));
        } 

        // Get Trigram theta
        if (trigramMap.get(lastLastWord + " " + lastWord + " " + token) != null) {
            double numerator = (double) trigramMap.get(
                lastLastWord + " " + lastWord + " " + token);
            px += l3 * (numerator / bigramMapWithStart.get(lastLastWord + " " + lastWord));
        } 

        return px;
    }

    // Create UNKs
    public static void storeUNKsFromUnigram() {
        unigramMap.putIfAbsent("UNK", 0);
        keysToUNK = new HashSet<String>();
        
        unigramMap.forEach((k,v) -> {
            if (v < 3) { // change the integer to change min_count for unk
                unigramMap.put("UNK", unigramMap.get("UNK") + v);
                keysToUNK.add(k);
            }
        });

        keysToUNK.forEach((k) -> {
            unigramMap.remove(k);
        });
    }

    // Calculate and print perplexity
    public static void reportPerplexity(Scanner s, String name) {
        // Unigram and bigram will always be valid because their history is seen
        int totalTokens = 0, numStops = 0;
        double logProbabilitySum = 0;
        String lastWord = "<START>", lastLastWord = null, token = "<START>";
        while (s.hasNextLine()) {
            numStops++; // Add stop at end of line
            String line = s.nextLine();
            Scanner lineScanner = new Scanner(line);

            do {
                // Advance
                lastLastWord = lastWord;
                lastWord = token;
                token = lineScanner.next();
                totalTokens++;

                double px = get_pX(lastLastWord, lastWord, token);
                logProbabilitySum += Math.log(px) / Math.log(2); // ln to log2

            } while (lineScanner.hasNext());

            // Advance for stop too
            lastLastWord = lastWord;
            lastWord = token;
            token = "\n";
            double px = get_pX(lastLastWord, lastWord, token);
            logProbabilitySum += Math.log(px) / Math.log(2); // ln to log2

        }
        
        // Add 1 stop and 2 starts per line for averaging calculations
        totalTokens += 3 * numStops;
        double averageLogProbability = logProbabilitySum / totalTokens;
        System.out.println(name + " perplexity is " + Math.pow(2.0, -averageLogProbability));
    }


    public static void makeScanners() throws FileNotFoundException {
        trainScanner = new Scanner(new File("./tokens/1b_benchmark.train.tokens"));
        devScanner = new Scanner(new File("./tokens/1b_benchmark.dev.tokens"));
        testScanner = new Scanner(new File("./tokens/1b_benchmark.test.tokens"));
    }
}