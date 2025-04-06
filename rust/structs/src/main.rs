fn main() {

    let mut user1 = User {
        active: true,
        username: String::from("Quison"),
        email: String::from("hej@gmail.com"),
        sign_in_count: 1,
    };

    let user1 = build_user(String::from("Viggo@viggoson.com"), String::from("viggo"));
    
    let user2 = User {
        email: String::from("Viggo2222@gssom.com"),
        ..user1
    };

    println!("{}", user1.email);
    println!("{}", user2.email);

    let black = Color(0,0,0);
    let origin = Point(0,0,0);

    let Color(r,g,b) = black;

    println!("r = {r}");

    let subject = AlwaysEqual;
}

fn build_user(email: String, username: String) -> User {
    User {
        active: true,
        username,
        email,
        sign_in_count: 1,
    }
}

struct User {
    active: bool,
    username: String,
    email: String,
    sign_in_count: u64,
}

struct Point(i32,i32,i32);
struct Color(i32,i32,i32);

struct AlwaysEqual;