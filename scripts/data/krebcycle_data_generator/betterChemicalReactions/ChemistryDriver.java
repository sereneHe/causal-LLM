package betterChemicalReactions;

import java.awt.Color;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;
import java.util.Scanner;
import java.nio.file.Paths;

/**
 * 
 * @author August Nagro
 *
 */
public class ChemistryDriver {

	private static final double size = 50;
	private static final double reactionRadius = 4;
	private static final double meanFreeTime = 2;
	private static final double temperature = 293.15;
	private static final double deltaTime = .01;
	private static ReactionsDictionary reactionDictionary = new ReactionsDictionary();
	private static ParticleContainer container;
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		challenge();
	}
	
	public static void level1(){
		container = new ParticleContainer(size, size, size, reactionRadius, reactionDictionary);
		RandomGenerator generator = new BoltzmannGenerator(container, temperature);
		HashMap<String, Double> catalysts = new HashMap<String, Double>();
		try {
			Scanner scan = new Scanner(new File("..\\commands\\level1.txt"));			
			parseReaction(scan, catalysts, container);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}		
		final int numPart = 1000;
		container.addRandomParticles(generator, numPart/2, "NH3");
		container.addRandomParticles(generator, numPart/2, "HCl");
		container.run(deltaTime);
	}
	public static void level2(){
		container = new ParticleContainer(size, size, size, reactionRadius, reactionDictionary);
		RandomGenerator generator = new BoltzmannGenerator(container, temperature);
		HashMap<String, Double> catalysts = new HashMap<String, Double>();
		try {
			Scanner scan = new Scanner(Paths.get("..", "commands", "level2.txt"));
			parseReaction(scan, catalysts, container);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}catch (IOException e) {
			e.printStackTrace();
		}
		final int numPart = 300;
		container.addRandomParticles(generator, numPart*2/3, "H2O");
		container.addRandomParticles(generator, numPart*1/3, "CO2");
		container.run(deltaTime);
	}
	
	public static void level3(){
		container = new ParticleContainer(size, size, size, reactionRadius, reactionDictionary);
		RandomGenerator generator = new BoltzmannGenerator(container, temperature);
		try {
			Scanner scan = new Scanner(new File("..\\commands\\level2.txt"));
			Scanner catScan = new Scanner(new File("..\\commands\\level2Catalyst.txt"));
			HashMap<String, Double> catalysts = parseCatalyst(catScan);
			parseReaction(scan, catalysts, container);
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		final int numPart = 300;
		container.addRandomParticles(generator, numPart*2/3, "H2O");
		container.addRandomParticles(generator, numPart*1/3, "CO2");
		container.addRandomParticles(generator, numPart/3, "CAH");
		container.run(deltaTime);
	}
	public static void challenge(){
		container = new ParticleContainer(size, size, size, reactionRadius, reactionDictionary);
		RandomGenerator generator = new BoltzmannGenerator(container, temperature);
		HashMap<String, Double> catalysts = new HashMap<String, Double>();
		try {
			Scanner scan = new Scanner(Paths.get("..", "commands", "challenge.txt"));
			parseReaction(scan, catalysts, container);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}

	}
	
	public static void parseReaction(Scanner scan, HashMap<String, Double> catalysts, ParticleContainer container){		
		do{
			HashMap<String, Integer> reactants = new HashMap<String, Integer>();
			HashMap<String, Integer> products = new HashMap<String, Integer>();
			boolean reversable = false;
			
			String line = scan.nextLine();
			String[] splitLine = new String[2];
			
			if(line.contains(" -> ")){
				splitLine = line.split(" -> ");
			}else if(line.contains(" <-> ")){
				reversable = true;
				splitLine = line.split(" <-> ");
			}
			
			String[] react = splitLine[0].split(" ");
			double forwardAE = Double.parseDouble(react[0]);
			double backwardsAE = Double.parseDouble(react[1]);			
			
			String[] prod = splitLine[1].split(" ");
			for(int i=2; i<react.length; i+=3){
				String name = react[i].substring(1);
				int numberPart = Integer.parseInt(react[i].substring(0, 1));
				if(reactants.containsKey(name)){
					reactants.put(name, reactants.get(name)+numberPart);
				}else{
					reactants.put(name, numberPart);
				}
				if(container.getDictionary().getList().contains(name) == false){
					double mass = Double.parseDouble(react[i+1]);
					Color color = parseColor(react[i+2]);
					container.getDictionary().addParticle(name, mass, color, meanFreeTime);
				}				
			}
			for(int i=0; i<prod.length; i+=3){
				String name = prod[i].substring(1);
				int numberPart = Integer.parseInt(prod[i].substring(0, 1));
				if(products.containsKey(name)){
					products.put(name, products.get(name)+numberPart);
				}else{
					products.put(name, numberPart);
				}
				if(container.getDictionary().getList().contains(name) == false){
					double mass = Double.parseDouble(prod[i+1]);
					Color color = parseColor(prod[i+2]);
					container.getDictionary().addParticle(name, mass, color, meanFreeTime);
				}
				
			}
			reactionDictionary.addReaction(reactants, products, reversable, catalysts, forwardAE, backwardsAE);
			
		}while(scan.hasNextLine());
	
	}
	public static HashMap<String, Double> parseCatalyst(Scanner catScan){
		HashMap<String, Double> catalysts = new HashMap<String, Double>();
		do{
			String[] line = catScan.nextLine().split(" ");
			String name = line[0];	
			double mass = Double.parseDouble(line[1]);
			Color color = parseColor(line[2]);
			double deltaAE = Double.parseDouble(line[3]);
			
			catalysts.put(name, deltaAE);
			
			if(container.getDictionary().getList().contains(name) == false){				
				container.getDictionary().addCatalyst(name, mass, color, meanFreeTime, deltaAE);
			}
			
		}while(catScan.hasNextLine());
		return catalysts;
	}
	public static Color parseColor(String color){
		switch (color) {
		case "RED": return Color.red;
		case "ORANGE": return Color.orange;
		case "YELLOW": return Color.yellow;
		case "GREEN": return Color.green;
		case "BLUE": return Color.blue;
		case "PINK": return Color.pink;
		default:
			return Color.WHITE;
		}
	}
}
