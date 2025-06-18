//3 Sum of numbers 1 to N (input from user)

import java.util.Scanner;
class Sum{
    public static void main(String[]args){
        Scanner sc = new Scanner(System.in);
        System.out.println("Enter the number: ");
        //receiving input from th user
        int num = sc.nextInt();
        //taking the initial value of sum as zero
        int sum=0;
        //using for loop and finding out the sum 
        for(int i=0;i<=num;i++){
            sum+=i;
        }
        System.out.println(sum);
        sc.close();
    }
}
