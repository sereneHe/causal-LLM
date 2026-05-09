package betterChemicalReactions;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Point;
import java.awt.Rectangle;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class Timeline implements Serializable {

	private static final long serialVersionUID = 2721365553382126027L;
	
	private String name;
	private Color color;
	private List<Double> timeline = new ArrayList<>();
	
	public Timeline(String name, Color color) {
		this.name = name;
		this.color = color;
	}
	
	public String getName() {
		return name;
	}
	
	public Color getColor() {
		return color;
	}
	
	public void addPoint(double population) {
		timeline.add(population);
	}
	
	public int getSize() {
		return timeline.size();
	}
	
	public double getMax() {
		double max = 0;
		for (Double ii : timeline) {
			if (ii > max) {
				max = ii;
			}
		}
		return max;
	}
	
	static private int circleRadius = 5;

	public void paint(Graphics g, Rectangle grid, double max) {
		g.setColor(getColor());
		
		double divisionSize = (double)grid.getWidth() / getSize();
		double currentLocation = grid.getMinX();
		Point formerLocation = new Point(0, 0);
		
		for (Double point : timeline) {
			int x = (int)Math.round(currentLocation);
			int y = (int)(grid.getMaxY() - point / max * grid.getHeight());

			g.fillOval(x - circleRadius, y - circleRadius, 2 * circleRadius, 2 * circleRadius);
			if (currentLocation != (int)grid.getMinX()) {
				g.drawLine((int)formerLocation.getX(), (int)formerLocation.getY(), x, y);
			}
			
			formerLocation.setLocation(x, y);
			currentLocation += divisionSize;
		}
	}
}
