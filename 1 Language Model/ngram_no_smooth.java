/*
 * Amol Sharma
 * This file implements simple uni, bi and trigram models without smoothing
 * Compile: javac part1a.java
 * Run: java part1a
 */
 
import java.io.*;
import java.util.*;

public class ngram_no_smooth {
    
    public static HashMap<String, Integer> unigramMap, bigramMap,
        trigramMap, currentMap;
    public static Scanner trainScanner;
    public static Scanner devScanner;
    public static Scanner testScanner;
    public static int totalTokensInTrainingFile;
    public static HashSet<String> keysToUNK;
    public static int historyLength;
    
    public static void main(String[] args) {
        try {
            unigramMap = new HashMap<String, Integer>(30000);
            bigramMap = new HashMap<String, Integer>(30000);
            trigramMap = new HashMap<String, Integer>(30000);

            // For each n-gram
            for (int i = 1; i < 4; i++) {
                historyLength = i;
                totalTokensInTrainingFile = 0;
                train();
                test();
            }

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

    // Reset file readers and determine perplexity based on those files
    public static void test() throws FileNotFoundException {
        makeScanners();
        reportPerplexity(trainScanner, "Training");
        reportPerplexity(devScanner, "Dev");
        reportPerplexity(testScanner, "Test");
    }

    // Read the file and count the occurences of the tokens. Put them into appropriate maps
    public static void readTrainingFile() throws FileNotFoundException {
        int numStops = 0;
        currentMap.putIfAbsent("\n", 0); // Stops

        String lastWord = null, lastLastWord = null, token = null;

        while (trainScanner.hasNextLine()) { // For each line
            String line = trainScanner.nextLine();
            Scanner lineScanner = new Scanner(line); // Scan the line

            while (lineScanner.hasNext()) { // For each word
                // Get the next word and keep track of the last two
                lastLastWord = lastWord;
                lastWord = token;
                token = lineScanner.next();
                
                totalTokensInTrainingFile++;
        
                // Increment count in the appropriate map
                insertCounts(lastLastWord, lastWord, token);
            }

            // Advance once more for stop symbol at the end of the line
            lastLastWord = lastWord;
            lastWord = token;
            token = "\n"; // This is the stop symbol that I chose
            insertCounts(lastLastWord, lastWord, token);
            
            lineScanner.close();
            numStops++;
        }
        
        // Include stops in token count
        totalTokensInTrainingFile += numStops;

        // UNK based on unigrams
        if (historyLength == 1) storeUNKsFromUnigram();
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
                if (lastWord != null) { // Valid biagram
                    // UNK words
                    if (keysToUNK.contains(lastWord)) {lastWord = "UNK";}
                    if (keysToUNK.contains(token)) {token = "UNK";}

                    // Add current word combination
                    String combination = lastWord + " " + token;
                    bigramMap.putIfAbsent(combination, 0);
                    bigramMap.put(combination, bigramMap.get(combination) + 1);   
                }
                break;
            case 3:
                if (lastLastWord != null && lastWord != null) {
                    if (keysToUNK.contains(lastLastWord)) {lastLastWord = "UNK";}
                    if (keysToUNK.contains(lastWord)) {lastWord = "UNK";}
                    if (keysToUNK.contains(token)) {token = "UNK";}

                    // Add current word combination
                    String combination = lastLastWord + " " + lastWord + " " + token;
                    trigramMap.putIfAbsent(combination, 0);
                    trigramMap.put(combination, trigramMap.get(combination) + 1);
                }
                break;
            default:
                break;
        }
    }

    public static double get_pX(String lastLastWord, String lastWord,
        String token) {
        double px = 0.0;

        switch(historyLength) {
            case 1: // Unigram
                if (unigramMap.get(token) == null) { token = "UNK"; } // UNK if word is unseen
                px = (double)unigramMap.get(token) / totalTokensInTrainingFile;
                break;
            case 2: // Bigram
                if (lastWord != null) { // Valid bigram
                    if (unigramMap.get(lastWord) == null) {lastWord = "UNK";} 
                    if (unigramMap.get(token) == null) {token = "UNK";}
                    if (bigramMap.get(lastWord + " " + token) != null) {
                        double numerator = (double) bigramMap.get(lastWord + " " + token);
                        px = numerator / unigramMap.get(lastWord);
                    } 
                }
                break;
            case 3: // Trigram
                if (lastWord != null && lastLastWord != null) { // Valid trigram
                    if (unigramMap.get(lastLastWord) == null) {lastLastWord = "UNK";} 
                    if (unigramMap.get(lastWord) == null) {lastWord = "UNK";} 
                    if (unigramMap.get(token) == null) {token = "UNK";}
                    if (trigramMap.get(lastLastWord + " " + lastWord + " " + token) != null) {
                        double numerator = (double) trigramMap.get(
                            lastLastWord + " " + lastWord + " " + token);
                        px = numerator / bigramMap.get(lastLastWord + " " + lastWord);
                    } 
                }
                break;
            default:
                break;
        }

        return px;
    }

    // Create UNKs and remember unked words in keysToUNK
    public static void storeUNKsFromUnigram() {
        unigramMap.putIfAbsent("UNK", 0);
        keysToUNK = new HashSet<String>();
        
        unigramMap.forEach((k,v) -> {
            if (v < 3) {
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
        int totalTokens = 0;
        double logProbabilitySum = 0;
        String lastWord = null, lastLastWord = null, token = null;
        while (s.hasNextLine()) {
            totalTokens++; // Increment for stops
            String line = s.nextLine();
            Scanner lineScanner = new Scanner(line);

            while (lineScanner.hasNext()) {
                // Advance
                lastLastWord = lastWord;
                lastWord = token;
                token = lineScanner.next();
                totalTokens++;

                double px = 1; // log(1) is 0;
                // TODO:- below works but can be structured better
                if (historyLength == 2) {
                    // If we're on the first token in the line,
                    // but we're looking for bigrams, skip
                    if (lastWord != null) {
                        px = get_pX(lastLastWord, lastWord, token);
                    } else {
                        continue;
                    }
                } else if (historyLength == 3) {
                    // If we're on the first or second token in the line,
                    // but we're looking for trigrams, skip
                    if (lastLastWord != null && lastWord != null) {
                        px = get_pX(lastLastWord, lastWord, token);
                    } else {
                        continue;
                    }
                } else {
                    px = get_pX(lastLastWord, lastWord, token);
                }
                
                logProbabilitySum += Math.log(px) / Math.log(2); // ln to log2
                
            }

            // Advance for stop too
            lastLastWord = lastWord;
            lastWord = token;
            token = "\n";
            double px = get_pX(lastLastWord, lastWord, token);
            logProbabilitySum += Math.log(px) / Math.log(2); // ln to log2

        }

        double averageLogProbability = logProbabilitySum / totalTokens; // Divide by total tokens in current file
        System.out.println(name + " perplexity is " + Math.pow(2.0, -averageLogProbability));
    }


    public static void makeScanners() throws FileNotFoundException {
        trainScanner = new Scanner(new File("./tokens/1b_benchmark.train.tokens"));
        devScanner = new Scanner(new File("./tokens/1b_benchmark.dev.tokens"));
        testScanner = new Scanner(new File("./tokens/1b_benchmark.test.tokens"));
    }
}