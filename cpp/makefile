# define all subfolders used
IDIR = include/
ODIR = obj/
SDIR = src/

# define the filename containing the main function, and the name of the executable
EXECUTABLE = noisy_lpm
MAIN = main

# get all filenames for the sources and objects
SRC = $(notdir $(wildcard $(SDIR)*.cpp))
OBJS = $(ODIR)$(MAIN).o $(addprefix $(ODIR), $(SRC:.cpp=.o))

# compiler and flags
CC = mpic++
CFLAGS = -c -g -Wall -O2 -I$(IDIR)
LFLAGS = -lm -larmadillo# add additional external libraries here

# rule for the executable file
$(MAIN) : $(OBJS)
	$(CC) $(OBJS) -o $(EXECUTABLE) $(LFLAGS)

# rule for the source file containing the main function
$(ODIR)$(MAIN).o : $(MAIN).cpp
	$(CC) $(CFLAGS) $< -o $@

# rule for all other source files in the source folder
$(ODIR)%.o : $(SDIR)%.cpp
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f *.o $(ODIR)*.o $(EXECUTABLE)

### Run:
### mpirun --bind-to core --nooversubscribe -np 2 --report-bindings --display-allocation --display-map noisy_lpm


