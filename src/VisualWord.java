import java.util.List;

import mpi.cbg.fly.Feature;


public class VisualWord {
	
	//the Cebir FeatureVector
	public Feature centroied;
	
	//the unique class ID
	public int	classID;
	
	//a placeholder for a class verification value
	public Object verificationValue;		
	
	//list of near features to a centeroid
	public List<Feature> nearFeature;

}
