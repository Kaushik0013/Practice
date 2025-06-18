//4 Factorial of a number using loop

import java.util.Scanner;

class Factorial{
    public static void main(String[]args){
        Scanner sc = new Scanner(System.in);
        System.out.println("Enter the number: ");
        //Receiving inout from the user
        int num= sc.nextInt();
        //initial value of factorial var is 1 since anything multiplied with 0 will be 0
        int factorial = 1;
        //using for loop for finding the factorial of the number received
        for(int i = num; i>=1; i--){
            factorial*=i;
        }
        System.out.println(factorial);
        sc.close();
    }
}