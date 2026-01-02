from core_components.agents.agent_00 import explain_like_5

def main():
    topic = "Quantum Computing"
    print(f"Explanation of '{topic}':")
    
    # Since explain_like_5 is now a generator, we loop over it here
    for chunk in explain_like_5(topic):
        print(chunk, end="", flush=True)

if __name__ == "__main__":
    main()