/*	
 *	Example taken from gtk's website and used as a basis
 *	https://www.gtk.org/docs/getting-started/hello-world/
 *	
 *	
 *	
 *	
 *	
 */

#include <gtk/gtk.h>



static void print_hello (GtkWidget *widget, gpointer data){
	g_print("Hello from the terminal\n");	// statement printed to terminal
}

static void activate (GtkApplication *app, gpointer user_data){
	GtkWidget *window;	// instantiate window
	GtkWidget *button;	// instantiate button

	window = gtk_application_window_new (app);								// link my window to the application
	gtk_window_set_title(GTK_WINDOW (window), "test project");				// window title
	gtk_window_set_default_size(GTK_WINDOW (window), 200, 200);				// dimensions of window preset

	button = gtk_button_new_with_label("Click here for a surprise");		// create a button with a label of inserted text
	g_signal_connect(button, "clicked", G_CALLBACK (print_hello), NULL);	// event driver linkage between button on click to a function
	gtk_window_set_child(GTK_WINDOW (window), button);						// set button as a child of this specific window maybe?

	gtk_window_present(GTK_WINDOW (window));								// display this window
}

int main(int argc, char **argv){
	/*
	 *	create a GTKapplication object and run it.
	 *	here GTK application is created & initialize using
	 *	the gtk_application_new() function.
	 */ 
	GtkApplication *app;
	int status;

	app = gtk_application_new("com.github.acard6.MyApp", G_APPLICATION_DEFAULT_FLAGS);	// application ID
	g_signal_connect(app, "activate", G_CALLBACK (activate), NULL);						// linkage between app and activate function on activate
	status = g_application_run(G_APPLICATION (app), argc, argv);
	g_object_unref(app);

	return status;
}