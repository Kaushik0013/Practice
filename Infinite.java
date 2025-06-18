//5 Infinite loop with exit condition (type exit to stop)

import java.util.Scanner;

class Infinite{
    public static void main(String[]args){
        Scanner sc = new Scanner(System.in);
         // Infinite loop
        while(true){
            System.out.print("Enter the input(type 'exit' to stop): ");
            String input = sc.nextLine();

             // Check if input is "exit" (case-insensitive)
            if(input.equalsIgnoreCase("exit")){
                System.out.println("Exiting...");
                break;
            }
            // Otherwise, print the input back to the user
            System.out.println("You entered: " + input);
        }

        sc.close();
    }
}