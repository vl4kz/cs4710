// Feel free to use this java file as a template and extend it to write your solver.
// ---------------------------------------------------------------------------------

import world.Robot;
import world.World;

import java.awt.*;
import java.util.*;
import java.lang.Math;

public class MyRobot extends Robot {
    boolean isUncertain;
    public static final double DIAG_COST = 1.0;
    public static final double ADJ_COST = 1.0;
    public static Point END_POS = null;

    ArrayList<Point> closedList;
    PriorityQueue<Point> openQueue;

    HashMap<Point, Double> fnMap; // stores f(n) = g(n) + h(n)
    HashMap<Point, Double> gnMap; // stores g(n) -- the cost of the best path so far to point n
    HashMap<Point, Point> parentMap; // stores the parent cell of each point

    @Override
    public void travelToDestination() {
        if (isUncertain) {
			// call function to deal with uncertainty
        }
        else {
			// call function to deal with certainty

      // INITIALIZE THE OPEN AND CLOSED LISTS
      // instance of a new PriorityQueue where the head is points with LOWER heuristic values (less costly values)
      openQueue = new PriorityQueue<Point>(10, new Comparator<Point>() {
        public int compare(Point p1, Point p2) {
          if (fnMap.getOrDefault(p1, Double.POSITIVE_INFINITY) < fnMap.getOrDefault(p2, Double.POSITIVE_INFINITY)) {
            return -1;
          }
          else if (fnMap.getOrDefault(p1, Double.POSITIVE_INFINITY) > fnMap.getOrDefault(p2, Double.POSITIVE_INFINITY)) {
            return 1;
          }
          else {
            return 0;
          }
        }
      }
      );

      closedList = new ArrayList<Point>();
      fnMap = new HashMap<Point, Double>();
      gnMap = new HashMap<Point, Double>();
      parentMap = new HashMap<Point, Point>();

      // PLACE THE STARTING POINT INTO THE QUEUE
      Point startingPoint = new Point(super.getX(), super.getY());
      openQueue.add(startingPoint);
      fnMap.put(startingPoint, diagDistance(startingPoint, END_POS));
      gnMap.put(startingPoint, 0.0);

      while (openQueue.size() != 0) {  // while the queue is not empty
        Point currentPoint = openQueue.poll(); // continue popping off

        if (currentPoint.equals(END_POS)) {
          findPath(parentMap, currentPoint);
          break;
        }

        closedList.add(currentPoint);
        // System.out.println(currentPoint);

        // iterate through all possible diagonal directions for neighbors
          for (int rowOffset = -1; rowOffset <= 1; rowOffset++) {
            for (int colOffset = -1; colOffset <= 1; colOffset++) {

              Point neighbor = new Point((int) currentPoint.getX() + rowOffset, (int) currentPoint.getY() + colOffset); // create a new possible point to move to

              // System.out.println(neighbor + " " + String.valueOf(rowOffset) + " " + String.valueOf(colOffset) + " " + super.pingMap(neighbor));

              if (super.pingMap(neighbor) != null && !(super.pingMap(neighbor).equals("X"))) { // check if the point is within the boundaries of the world and not a wall
                // System.out.println(neighbor);

                if (closedList.contains(neighbor)) {
                  continue; // this point has been already evaluated
                }

                if (!(openQueue.contains(neighbor))) { // if the neighbor is not in the openQueue
                  openQueue.add(neighbor);
                }

                double movementDistance = diagDistance(neighbor, currentPoint); // cost of moving from currentPoint to neighbor

                // calculate the new g(n) by adding the old value of g(n) + cost of moving points
                double newgn = gnMap.getOrDefault(currentPoint, Double.POSITIVE_INFINITY) + movementDistance;

                if (newgn < gnMap.getOrDefault(neighbor, Double.POSITIVE_INFINITY)) {
                  parentMap.put(neighbor, currentPoint);
                  gnMap.put(neighbor, newgn);
                  double heuristicCost = diagDistance(neighbor, END_POS);
                  fnMap.put(neighbor, gnMap.get(neighbor) + heuristicCost);
                }
              }
            }
          }
      }


        }
    }

    @Override
    public void addToWorld(World world) {
        isUncertain = world.getUncertain();
        super.addToWorld(world);
    }


    public double diagDistance(Point p1, Point p2) {
      double dx = Math.abs(p1.getX() - p2.getX());
      double dy = Math.abs(p1.getY() - p2.getY());

      return ADJ_COST * (dx + dy) + (DIAG_COST - 2.0 * ADJ_COST) * Math.min(dx, dy);
    }

    public void findPath(HashMap<Point, Point> parentMap, Point currentPoint) {
      ArrayList<Point> totalPath = new ArrayList<>();
      totalPath.add(currentPoint);

      while (parentMap.keySet().contains(currentPoint)) {
        currentPoint = parentMap.get(currentPoint);
        totalPath.add(currentPoint);
      }

      for (int i = totalPath.size()-1; i >= 0; i--) {
        System.out.println(totalPath.get(i));
      }
    }

    public static void main(String[] args) {
        try {
			World myWorld = new World("TestCases/myInputFile3.txt", false);

            MyRobot robot = new MyRobot();
            robot.addToWorld(myWorld);
            END_POS = myWorld.getEndPos();
			//myWorld.createGUI(400, 400, 200); // uncomment this and create a GUI; the last parameter is delay in msecs


			robot.travelToDestination();
        }

        catch (Exception e) {
            e.printStackTrace();
        }
    }
}
