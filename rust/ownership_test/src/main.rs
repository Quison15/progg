fn main() {
    let mut s: String = String::from("hello");
    s.push_str(", world!");
    println!("{s}");

    let x = 5;
    let y = x;

    let s1 = String::from("hello");
    let s2 = s1;

    println!("{s2}, world!");

    let s1 = String::from("Hello");

    let len = calculate_length(&s1);

    println!("The length of '{s1}' is {len}");

    let mut s = String::from("Hello");

    println!("{s}");

    change(&mut s);

    println!("{s}");

}

fn change(some_string: &mut String) {
    some_string.push_str(" World!");
}

fn calculate_length(s: &String) -> usize {
    s.len()
}
