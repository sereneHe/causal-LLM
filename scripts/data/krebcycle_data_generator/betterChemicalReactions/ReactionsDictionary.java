package betterChemicalReactions;

import betterChemicalReactions.ReactionsDictionary.ReactionInfo;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

public class ReactionsDictionary implements Iterable<ReactionInfo> {

    @Override
    public Iterator<ReactionInfo> iterator() {
        return reactions.iterator();
    }

    public class ReactionInfo {

        HashMap<String, Integer> reactants;
        HashMap<String, Integer> products;
        HashMap<String, Double> catalysts;
        double forewardAE;
        double backwardAE;
        boolean reversable;

        public Map<String, Integer> getReactants() {
            return Collections.unmodifiableMap(reactants);
        }

        public Map<String, Integer> getProducts() {
            return Collections.unmodifiableMap(products);
        }
        
        
    }

    private static ArrayList<ReactionInfo> reactions = new ArrayList<ReactionInfo>();
    private ArrayList<Particle> particlesToRemove = new ArrayList<Particle>();

    public void addReaction(HashMap<String, Integer> reactants, HashMap<String, Integer> products, boolean reversable, HashMap<String, Double> catalysts, double forewardAE, double backwardAE) {
        ReactionInfo info = new ReactionInfo();
        info.reactants = reactants;
        info.products = products;
        info.reversable = reversable;
        info.catalysts = catalysts;
        info.forewardAE = forewardAE;
        info.backwardAE = backwardAE;
        reactions.add(info);

    }

    public double getAE(Particle p) {
        for (ReactionInfo r : reactions) {
            for (String react : r.reactants.keySet()) {
                if (p.getName().equals(react)) {
                    return r.forewardAE;
                }
            }
            for (String prod : r.products.keySet()) {
                if (p.getName().equals(prod)) {
                    return r.backwardAE;
                }
            }
        }
        return 0;
    }

    public HashMap<String, Double> getCatalysts(Particle p) {
        for (ReactionInfo r : reactions) {
            for (String react : r.reactants.keySet()) {
                if (p.getName().equals(react)) {
                    return r.catalysts;
                }
            }
        }
        return new HashMap<String, Double>();
    }

    public HashMap<String, Integer> reactionResults(List<Particle> list) {
        particlesToRemove = new ArrayList<Particle>();
        HashMap<String, Integer> newParticles = new HashMap<String, Integer>();

        //sorts inputed list
        HashMap<String, ArrayList<Particle>> sortedList = new HashMap<String, ArrayList<Particle>>();
        for (Particle p : list) {
            if (sortedList.containsKey(p.getName()) && sortedList.get(p.getName()) != null) {
                ArrayList<Particle> temp = sortedList.get(p.getName());
                temp.add(p);
                sortedList.put(p.getName(), temp);
            } else {
                ArrayList<Particle> temp = new ArrayList<Particle>();
                temp.add(p);
                sortedList.put(p.getName(), temp);
            }
        }
        //checks if reaction possible
        for (ReactionInfo r : reactions) {

            if (r.reactants.containsKey(list.get(0).getName())) {
                boolean enoughParticles = true;
                for (String key : r.reactants.keySet()) {
                    if (sortedList.containsKey(key) == false) {
                        enoughParticles = false;
                    } else if (sortedList.get(key).size() < r.reactants.get(key)) {
                        enoughParticles = false;
                    }

                }
                if (enoughParticles) {
                    newParticles = r.products;
                    for (String key : r.reactants.keySet()) {
                        int numParticles = r.reactants.get(key);
                        int index = 0;
                        while (numParticles > 0) {
                            particlesToRemove.add(sortedList.get(key).get(index));
                            numParticles--;
                            index++;
                        }
                    }
                }
            }
            if (r.reversable && r.products.containsKey(list.get(0).getName())) {
                boolean enoughParticles = true;
                for (String key : r.products.keySet()) {
                    if (sortedList.containsKey(key) == false) {
                        enoughParticles = false;
                    } else if (sortedList.get(key).size() < r.products.get(key)) {
                        enoughParticles = false;
                    }
                }
                if (enoughParticles) {
                    newParticles = r.reactants;
                    for (String key : r.products.keySet()) {
                        int numParticles = r.products.get(key);
                        int index = 0;
                        while (numParticles > 0) {
                            particlesToRemove.add(sortedList.get(key).get(index));
                            numParticles--;
                            index++;
                        }
                    }
                }
            }
        }
        return newParticles;
    }

    public ArrayList<Particle> getParticlesToRemove() {
        return particlesToRemove;
    }

}
