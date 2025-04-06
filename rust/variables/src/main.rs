fn main() {
    let x = 5;
    println!("x = {x}");
    let x=6;
    println!("x = {x}");


    let x = 5;

    let x = x + 1;

    {
        let x = x * 2;
        println!("x = {x}");
    }

    println!("x= {x}");

    let spaces = "    ";
    let spaces: usize = spaces.len();
    println!("{spaces}");

    let tup = (500,6.4,1);

    let (x,y,z) = tup;
    println!("y = {y}");
    
    let y = {
        let x = 5;
        x+1
    };
    println!("{y}");
    let x = five();
    println!("{x}");

    let x = false;

    if x {
        println!("Truee");
    }

    let mut counter = 0;

    let result = loop {
        counter += 1;

        if counter == 10 {
            break counter * 2;
        }
    };
    println!("The result is {result}");

    let a = [10,20,30,40,50];
    for element in a {
        println!("The value is: {element}");
    }

    for number in (1..4).rev() {
        println!("{number}");
    }
    println!("LIFTOFFFFFFFFFFFFF!!!");
}

fn five() -> i32 {
    return 5;
}
