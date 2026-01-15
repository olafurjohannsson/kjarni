#!/bin/bash
mkdir -p rag_test_data

# ==========================================
# CASE 1: "Apple" (The classic ambiguity)
# ==========================================
# Query: "Apple stock price"
# ------------------------------------------
# Distractor 1: Has "Apple", "Stock", "Price" keywords but wrong context (Food)
echo "The price of apple stock for making cider has risen. Farmers are happy." > rag_test_data/apple_cider.txt
# Distractor 2: Tech related but wrong specific intent
echo "Apple released a new iPhone with a high price tag. Stocks are limited." > rag_test_data/apple_iphone.txt
# TARGET: The actual financial news
echo "Apple Inc. (AAPL) shares closed higher today as the tech giant reported earnings." > rag_test_data/apple_finance.txt

# ==========================================
# CASE 2: "Rust" (Language vs Metal)
# ==========================================
# Query: "How to handle errors in Rust?"
# ------------------------------------------
# Distractor 1: Metal rust (high keyword match on 'rust' and 'error/failure')
echo "Structural errors in bridges are often caused by rust corrosion on metal beams." > rag_test_data/rust_metal.txt
# Distractor 2: Rust game
echo "In the game Rust, handling server errors requires a restart." > rag_test_data/rust_game.txt
# TARGET: Programming
echo "In Rust programming, the Result<T, E> type is used for recoverable errors. Use match or ? operator." > rag_test_data/rust_code.txt

# ==========================================
# CASE 3: Negation (Vectors are bad at this)
# ==========================================
# Query: "What is NOT a fruit?"
# ------------------------------------------
# Distractor: Mentions fruit
echo "An apple is a fruit." > rag_test_data/fruit_apple.txt
echo "A banana is a fruit." > rag_test_data/fruit_banana.txt
# TARGET: 
echo "A car is a vehicle, it is not a fruit." > rag_test_data/not_fruit.txt

echo "✅ Test data generated in 'rag_test_data/'"
