use std::thread;
use std::io::Write;
use std::net::TcpListener;

fn main() {
    let listener = TcpListener::bind("127.0.0.1:8080").unwrap();
    println!("listening started, ready to accept");

    // accept connections and process them, spawning a new thread for each one
    for stream in listener.incoming() {
        thread::spawn(|| {
            let mut stream = stream.unwrap();
            stream.write(b"Hello World\r\n").unwrap();
        });
    }
}
