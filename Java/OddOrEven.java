import java.util.Scanner;

/* Exempel pÃ¥ ett program:
 * Det mÃ¥ste finnas en main-metod. Se de tvÃ¥ fÃ¶rsta och sista
 * raderna som en "ram" runt satserna som exekveras. Programmets
 * namn (hÃ¤r OddOrEven) bestÃ¤mmer du sjÃ¤lv.
 * 
 * { } omsluter raderna i ett block
 * 
 * Variablers typer mÃ¥ste deklareras. Typerna Ã¤r av tvÃ¥ slag; 
 * - primitiva datatyper (ex: int, double, boolean, char)
 * - referens, d.v.s klass eller vektor (ex: Scannner)
 */

public class OddOrEven {
	public static void main(String[] args) {
		System.out.println("Skriv ett heltal: ");
		Scanner scan = new Scanner(System.in);
		long n = scan.nextLong();
		boolean odd = n % 2 != 0;
		System.out.println(odd);
		if (odd) {
			System.out.println(n + " Ã¤r ett udda tal.");
		} else {
			System.out.println(n + " Ã¤r ett jÃ¤mnt tal.");
		}
	}
}