#gcc $(pkg-config --cflags gtk4) -o hello-world-gtk hello-world-gtk.c $(pkg-config --libs gtk4)

# courtesy of the people at stack overflow https://stackoverflow.com/questions/21548464/how-to-write-a-makefile-to-compile-a-simple-c-program

# variables
CC      = gcc			# compiler gcc for c++ / 
CFLAGS  = $(shell pkg-config --cflags gtk4)	# compiler and linker flags
LDFLAGS   = $(shell pkg-config --libs gtk4)

SRCS	= interface.c
OBJS 	= $(SRCS:.c-.o)
TARGET 	= interface			#output exe name

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCS) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -c $@

clean:
	rm -f $(TARGET) $(OBJS)