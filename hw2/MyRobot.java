// Feel free to use this java file as a template and extend it to write your solver.
// ---------------------------------------------------------------------------------

import world.Robot;
import world.World;

import java.awt.*;
import java.util.*;
import java.lang.Math;

public class MyRobot extends Robot {
    boolean isUncertain;
    public static final double DIAG_COST = 1.4142135623;
    public static final double ADJ_COST = 1.0;
    public static Point END_POS = null;
    public static final int UNCERTAIN_DIST = 2;

    ArrayList<Point> closedList;
    PriorityQueue<Point> openQueue;

    HashMap<Point, Double> fnMap; // stores f(n) = g(n) + h(n)
    HashMap<Point, Double> gnMap; // stores g(n) -- the cost of the best path so far to point n
    HashMap<Point, Point> parentMap; // stores the parent cell of each point
    HashSet<Point> blocked; // stores points that lead to blocked paths

    public MyRobot() {
        initRobot();
        blocked = new HashSet<Point>();
    }

    public void initRobot() {
        closedList = new ArrayList<Point>();
        fnMap = new HashMap<Point, Double>();
        gnMap = new HashMap<Point, Double>();
        parentMap = new HashMap<Point, Point>();

        // instance of a new PriorityQueue where the head is points with LOWER heuristic values (less costly values)
        openQueue = initQueue();
    }

    @Override
    public void travelToDestination() {
        // PLACE THE STARTING POINT INTO THE QUEUE
        Point startingPoint = this.getMyPoint();
        openQueue.add(startingPoint);
        fnMap.put(startingPoint, diagDistance(startingPoint, END_POS));
        gnMap.put(startingPoint, 0.0);

        if (isUncertain) {
            travelUncertain();
        }
        else {
            travelCertain();
        }
    }

    public void travelUncertain() {
        // 1. poll until distance 6 is reached
        // 2. Move robot to best distance
        // 3. reset the Maps DS
        HashSet<Point> blocked = new HashSet<>(); // Points in paths we've decided won't work
        Stack<Point> totalPath = new Stack<>();
        Point startPoint = this.getMyPoint();

        while (true) {  // while the queue is not empty
//            System.out.println("Current position of the robot: " + String.valueOf(this.getX()) + "," + String.valueOf(this.getY()));
            System.out.println(openQueue.isEmpty());
            if (openQueue.isEmpty()) {
                for (Point p : closedList) {
                    blocked.add(p);
                }
                backtrack(totalPath);

                initRobot();
                startPoint = this.getMyPoint();
                openQueue.add(startPoint);
                fnMap.put(startPoint, diagDistance(startPoint, END_POS));
                gnMap.put(startPoint, 0.0);
                continue;
            }

            Point currPoint = openQueue.poll(); // continue popping off
            System.out.println(currPoint);
            if (isDistanceGreater(UNCERTAIN_DIST, currPoint, startPoint) || currPoint.equals(END_POS) || areNeighborsBlocked(currPoint)) {
                // Found best point up to 6 away, or end point
                ArrayList<Point> path = findPath(parentMap, currPoint);
                ArrayList<Point> walkedSegment = moveRobot(path);
                for (Point p : walkedSegment) {
                    totalPath.add(p);
                }
                boolean moveSuccess = walkedSegment.size() == path.size();
                if (moveSuccess && currPoint.equals(END_POS)) {
                    // Made it to the end
                    return;
                } else if (!moveSuccess){
                    // if surroundings are all blocked, move back and block this point
                    if (areNeighborsBlocked(this.getMyPoint())) {
                        System.out.println("BACKTRACKING");
                        // call backtracking function here
                        backtrack(totalPath);

                        initRobot();

                        startPoint = this.getMyPoint();
                        openQueue.add(startPoint);
                        fnMap.put(startPoint, diagDistance(startPoint, END_POS));
                        gnMap.put(startPoint, 0.0);
                        continue;
                    } else {
                        // else restart a* from this point
                        System.out.println("PATH Segment: " + path);
                        initRobot();
                        startPoint = this.getMyPoint();
                        System.out.println("YOU HIT A BLOCK; RESTART A*: " + startPoint);
                        openQueue.add(startPoint);
                        fnMap.put(startPoint, diagDistance(startPoint, END_POS));
                        gnMap.put(startPoint, 0.0);
                        continue;
                    }
                } else {
                    // Restart A* algo from the current point
                    initRobot();
                    startPoint = this.getMyPoint();
                    System.out.println("OUTSIDE ELSE LOOP: " + startPoint);
                    openQueue.add(startPoint);
                    fnMap.put(startPoint, diagDistance(startPoint, END_POS));
                    gnMap.put(startPoint, 0.0);
                    continue;
                }
            }
            closedList.add(currPoint);
            processNeighbors(currPoint, true);
            System.out.println("SIZE: " + openQueue.size());
        }
    }

    public void travelCertain() {
        while (openQueue.size() != 0) {  // while the queue is not empty
            Point currentPoint = openQueue.poll(); // continue popping off

            if (currentPoint.equals(END_POS)) {
                ArrayList<Point> path = findPath(parentMap, currentPoint);
                moveRobot(path);
                return;
            }

            closedList.add(currentPoint);

            processNeighbors(currentPoint, false);
        }
    }

    public void backtrack(Stack totalPath) {

      do {
        System.out.println("HELLO");
        System.out.println(totalPath.peek());
        System.out.println(this.getMyPoint());
        blocked.add((Point) totalPath.peek());
        totalPath.pop();
        this.move((Point) totalPath.peek());
    } while (areNeighborsBlocked(this.getMyPoint()));
      /*
      blocked.add(totalPath.get(totalPath.size()-1));
      totalPath.remove(totalPath.size()-1);
      this.move(totalPath.get(totalPath.size()-1)); */
    }

    public Point getMyPoint() {
        return new Point(this.getX(), this.getY());
    }

    public void processNeighbors(Point currentPoint, boolean uncertain) {
        // iterate through all possible diagonal directions for neighbors
        for (int rowOffset = -1; rowOffset <= 1; rowOffset++) {
            for (int colOffset = -1; colOffset <= 1; colOffset++) {
                if (rowOffset == 0 && colOffset == 0) {
                    continue;
                }
                Point neighbor = new Point((int) currentPoint.getX() + rowOffset, (int) currentPoint.getY() + colOffset); // create a new possible point to move to

                if (super.pingMap(neighbor) != null && !blocked.contains(neighbor)) { // check if the point is within the boundaries of the world and not a wall
                    boolean isX = super.pingMap(neighbor).equals("X");
                    if (closedList.contains(neighbor) || (isX && !uncertain)) { //
                        continue; // this point has been already evaluated
                    }

                    openQueue.add(neighbor);


                    double movementDistance = diagDistance(neighbor, currentPoint); // cost of moving from currentPoint to neighbor

                    // calculate the new g(n) by adding the old value of g(n) + cost of moving points
                    double newgn;
                    if (uncertain && isX) {
                        newgn = Double.MAX_VALUE;
                    } else {
                        newgn = gnMap.getOrDefault(currentPoint, Double.MAX_VALUE) + movementDistance;
                    }

                    if (newgn < gnMap.getOrDefault(neighbor, Double.MAX_VALUE)) {
                        parentMap.put(neighbor, currentPoint);
                        gnMap.put(neighbor, newgn);
                        double heuristicCost = diagDistance(neighbor, END_POS);
                        fnMap.put(neighbor, gnMap.get(neighbor) + heuristicCost);
                    }
                }
            }
        }
    }

    public boolean areNeighborsBlocked(Point p) {
        for (int rowOffset = -1; rowOffset <= 1; rowOffset++) {
            for (int colOffset = -1; colOffset <= 1; colOffset++) {
                Point test = new Point((int) p.getX() + rowOffset, (int) p.getY() + colOffset);
                if (!blocked.contains(test) && super.pingMap(test) != null) {
                    return false;
                }
            }
        }
        return true;
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

    public boolean isDistanceGreater(int distance, Point p1, Point p2) {
        boolean xGreater = Math.abs(p1.getX() - p2.getX()) >= distance;
        boolean yGreater = Math.abs(p1.getY() - p2.getY()) >= distance;
        return xGreater || yGreater;
    }

    public PriorityQueue<Point> initQueue() {
        return new PriorityQueue<Point>(10, new Comparator<Point>() {
            public int compare(Point p1, Point p2) {
                double p1Cost = fnMap.getOrDefault(p1, Double.MAX_VALUE);
                double p2Cost = fnMap.getOrDefault(p2, Double.MAX_VALUE);
                if (p1Cost < p2Cost) {
                    return -1;
                } else if (p1Cost > p2Cost) {
                    return 1;
                } else {
                    return 0;
                }
            }
        });
    }

    public ArrayList<Point> findPath(HashMap<Point, Point> parentMap, Point currentPoint) {
      ArrayList<Point> totalPath = new ArrayList<>();
      totalPath.add(currentPoint);

      while (parentMap.keySet().contains(currentPoint)) {
        currentPoint = parentMap.get(currentPoint);
        totalPath.add(currentPoint);
      }
      Collections.reverse(totalPath);
      return totalPath;
    }

    /**
     * Moves robot along specified path. Path should be a connected sequence of points
     * Returns the path actually followed
     */
    public ArrayList<Point> moveRobot(ArrayList<Point> path) {
        ArrayList<Point> walked = new ArrayList<>();
        for (Point p : path) {
            this.move(p);
            blocked.add(p);
            // Robot did not move to the next point in path - something's wrong
            if (p.getX() != this.getX() || p.getY() != this.getY()) {
                System.out.println("ERROR MOVING");
                break;
            }
            walked.add(p);
        }
        return walked;
    }

    public static void main(String[] args) {
        try {
			World myWorld = new World("TestCases/myInputFile3.txt", true);

            MyRobot robot = new MyRobot();
            robot.addToWorld(myWorld);
            END_POS = myWorld.getEndPos();
			myWorld.createGUI(400, 400, 1); // uncomment this and create a GUI; the last parameter is delay in msecs


			robot.travelToDestination();
        }

        catch (Exception e) {
            e.printStackTrace();
        }
    }
}
