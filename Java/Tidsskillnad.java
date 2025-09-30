import java.util.Scanner;

public class Tidsskillnad {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        // Läs in starttid (timme och minut)
        int startTimme = scanner.nextInt();
        int startMinut = scanner.nextInt();

        // Läs in sluttid (timme och minut)
        int slutTimme = scanner.nextInt();
        int slutMinut = scanner.nextInt();

        // Omvandla båda tiderna till minuter
        int startTotalMinuter = startTimme * 60 + startMinut;
        int slutTotalMinuter = slutTimme * 60 + slutMinut;

        // Räkna ut skillnad i minuter
        int skillnadMinuter = slutTotalMinuter - startTotalMinuter;

        // Omvandla tillbaka till timmar och minuter
        int timmar = skillnadMinuter / 60;
        int minuter = skillnadMinuter % 60;

        // Skriv ut resultatet utan ledtext
        System.out.println(timmar + " " + minuter);
    }
}
