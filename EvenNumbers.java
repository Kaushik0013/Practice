//2 Print even numbers 2 to 50 using while

class EvenNumbers{
    public static void main(String[]args){
        //taking 1 as the initial value 
        int i = 1;
        while(i<=50){
            //using if condition to make sure the output contains only even numbers
            if(i%2==0){
                System.out.println(i);
            }
            i++;
        }
    }
}