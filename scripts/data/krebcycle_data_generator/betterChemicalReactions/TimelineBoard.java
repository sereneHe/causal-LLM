package betterChemicalReactions;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Rectangle;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import javax.swing.JPanel;

public class TimelineBoard extends JPanel implements ActionListener {

	private List<Timeline> timelines = new ArrayList<>();
	
	private static final int X_SIZE = 1000;
	private static final int Y_SIZE = 600;
	
	private static final double BORDER = .05;
	
	private ParticleContainer grid;
	
	private double maximum = 0;
	private int maxNameLength = 0;
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 2894866921233873348L;

	/**
	 * @param grid - this must have been filled with Animals before calling this function
	 */
	public TimelineBoard(ParticleContainer grid) {
		this.grid = grid;
		setSize(X_SIZE, Y_SIZE);
		setBackground(Color.WHITE);
		
		fillTimelines();
	}
	
	private void fillTimelines() {
		for (String str : grid.getDictionary().getList()) {
			addTimeline(new Timeline(str, grid.getDictionary().getColor(str)));
		}
	}
	
	public void addTimeline(Timeline timeline) {
		timelines.add(timeline);
		if (timeline.getName().length() > maxNameLength) {
			maxNameLength = timeline.getName().length();
		}
	}
	
	@Override
	public void actionPerformed(ActionEvent arg0) {
		updateTimelines();
		repaint();
	}
	
	private void updateTimelines() {
		Map<String, Integer> mymap = grid.getParticleMap();	
		for (Entry<String, Integer> entry : mymap.entrySet()) {
			Timeline myTimeline = getTimeline(entry.getKey());
			myTimeline.addPoint(entry.getValue());
		}
		setMax();
	}
	
	private Timeline getTimeline(String name) {
		for (Timeline timeline : timelines) {
			if (timeline.getName() == name) {
				return timeline;
			}
		}

		return null;
	}

	@Override
	public void paint(Graphics arg0) {
		super.paint(arg0);

		Rectangle grid = new Rectangle((int)(BORDER * X_SIZE), (int)(BORDER * Y_SIZE), (int)(X_SIZE * (1 - 2 * BORDER)),
				(int)(Y_SIZE * (1 - 4 * BORDER)));

		for (Timeline timeline : timelines) {
			timeline.paint(arg0, grid, maximum);
		}

		drawAxes(arg0, grid);
		drawLegend(arg0);
	}

	private static int LETTER_SIZE = 15;
	private static int LETTER_HEIGHT = 20;
	
	private void drawLegend(Graphics arg0) {		
		int leftEdge = (int)(X_SIZE * (1 - 2 * BORDER) - LETTER_SIZE * maxNameLength);
		double topPosition = BORDER * Y_SIZE;
		arg0.setFont(new Font(arg0.getFont().getFontName(), Font.PLAIN, (int)(LETTER_HEIGHT * .9)));
		
		for (Timeline timeline : timelines) {
			arg0.setColor(timeline.getColor());
			arg0.drawString(timeline.getName(), leftEdge, (int)(topPosition + LETTER_HEIGHT));
			arg0.fillRect(leftEdge - 2 * LETTER_SIZE, (int)topPosition, LETTER_SIZE, LETTER_HEIGHT);
			topPosition += LETTER_HEIGHT;
		}	
	}
	
	private void drawAxes(Graphics arg0, Rectangle grid) {
		double increment = increment();
		arg0.setColor(Color.BLACK);
		arg0.drawLine((int)grid.getMinX(), (int)grid.getMinY(), (int)grid.getMinX(), (int)grid.getMaxY());
		
		for (double counter = 0; counter < maximum; counter += increment) {
			int positionY = (int)(grid.getMaxY() - counter / maximum * grid.getHeight());
			arg0.drawString(Integer.toString((int)counter), (int)grid.getMinX(), positionY);
		}
	}
	
	private double increment() {
		if (maximum == 0) {
			return 0;
		}
		int exponent = (int)Math.floor(Math.log10(maximum));
		int base = (int)Math.pow(10, exponent);
		int mantissa = (int)maximum / base;
		
		if (mantissa > 5) {
			return base;
		} else {
			return (double)base / 5;
		}
	}
	
	private void setMax() {
		for (Timeline timeline : timelines) {
			double checkMax = timeline.getMax();
			if (checkMax > maximum) {
				maximum = checkMax;
			}
		}
	}

}
