NAME = induction
SRC = src
CLASSES = classes
JAVA_HOME ?= /Library/Java/JavaVirtualMachines/adoptopenjdk-8.jdk/Contents/Home
JAVAC = $(JAVA_HOME)/bin/javac
JAR = $(JAVA_HOME)/bin/jar
SCALAC = JAVA_HOME="$(JAVA_HOME)" PATH="$(JAVA_HOME)/bin:$$PATH" scalac

compile:
	test -x "$(JAVAC)" || (echo "JAVA_HOME=$(JAVA_HOME) does not point to a usable JDK. Set JAVA_HOME to a JDK 8 install."; exit 1)
	mkdir -p $(CLASSES)
	$(JAVAC) -cp $(CLASSES) -d $(CLASSES) `find $(SRC) -name "*.java"`
	$(SCALAC) -cp $(CLASSES)  -d $(CLASSES) `find $(SRC) -name "*.scala"`
	$(JAR) cf $(NAME).jar -C $(CLASSES) .
	$(JAR) uf $(NAME).jar -C $(SRC) .

clean:
	rm -rf $(CLASSES) $(NAME).jar
