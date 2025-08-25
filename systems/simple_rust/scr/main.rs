use serde::Deserialize;
use std::error::Error;
use std::fs::File;

// Define a struct for each CSV row
#[derive(Debug, Deserialize)]
struct Bar {
    date: String,
    close: f64,
}

// Calculate simple moving average
fn sma(data: &[f64], window: usize) -> Vec<f64> {
    data.windows(window)
        .map(|w| w.iter().sum::<f64>() / window as f64)
        .collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    // Open CSV file (you can change path later)
    let file = File::open("prices.csv")?;
    let mut rdr = csv::Reader::from_reader(file);

    // Collect closing prices
    let mut closes: Vec<f64> = Vec::new();
    for result in rdr.deserialize() {
        let record: Bar = result?;
        closes.push(record.close);
    }

    // Compute two SMAs
    let short = sma(&closes, 5);
    let long = sma(&closes, 20);

    // Generate simple crossover signals
    for i in (long.len().saturating_sub(short.len()))..short.len() {
        if short[i] > long[i] {
            println!("Day {}: BUY", i);
        } else if short[i] < long[i] {
            println!("Day {}: SELL", i);
        } else {
            println!("Day {}: HOLD", i);
        }
    }

    Ok(())
}

