NAME = induction
SRC = src
CLASSES = classes

compile:
	mkdir -p $(CLASSES)
	javac -cp $(CLASSES) -d $(CLASSES) `find $(SRC) -name "*.java"`
	scalac -cp $(CLASSES)  -d $(CLASSES) `find $(SRC) -name "*.scala"`
	jar cf $(NAME).jar -C $(CLASSES) .
	jar uf $(NAME).jar -C $(SRC) .

clean:
	rm -rf $(CLASSES) $(NAME).jar
