JFLAGS = -g
JC = javac
JAR = -cp ".:HW2.jar"

.SUFFIXES: .java .class
.java.class:
	$(JC) $(JFLAGS) $(JAR) $*.java

CLASSES = \
	MyRobot.java

default: classes

classes: $(CLASSES:.java=.class)

run:
	@java $(JAR) MyRobot

clean:
	$(RM) *.class
