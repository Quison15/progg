import java.util.Scanner;

/* Exempel med statiska metoder. 
 * Statiska metoder anropas Klassnamnn.metodnamn(argumentlista).
 * Metodens returtyp mÃ¥ste anger (hÃ¤r void eftersom inget
 * vÃ¤rde returneras.
 */

public class Asterisks {
	public static void main(String[] args) {
		System.out.println("Antal: ");
		Scanner scan = new Scanner(System.in);
		int nbr = scan.nextInt();
		Asterisks.printAsterisks(nbr);
		// eller eftersom printAsterisks finns i samma klass:
		// printAsterisks(n);
	}
	
	public static void printAsterisks(int n) {
		for (int  i = 0; i < n; i++) {
			System.out.print('*');
		}
		System.out.println();
	}
}