from vision import detect_objects
from llm import generate_text

def main():
    img = input("Image path: ").strip()
    prompt = input("Text prompt: ").strip()

    try:
        items = detect_objects(img)
        summary = "Detected objects:\n" + "\n".join(f"- {lbl}: {conf:.2f}" for lbl, conf in items) \
            if items else "No objects detected."
        print("\n" + summary + "\n")

        response = generate_text(summary, prompt)
        print("=== Model Response ===")
        print(response)

    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()


