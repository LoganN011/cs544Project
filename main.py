import argparse
import sys
from capture import capture_new_user
from train import train_model
from recognize import recognize_users

def interactive_mode():
    while True:
        print("\n=== Face Recognition & Liveness System ===")
        print("1. Capture new user")
        print("2. Train model")
        print("3. Recognize users")
        print("4. Exit")
        choice = input("Select an option (1-4): ")

        if choice == '1':
            name = input("Enter user name: ").strip()
            if not name:
                print("Name cannot be empty.")
                continue
            samples_str = input("Enter number of samples (default 100): ").strip()
            samples = int(samples_str) if samples_str.isdigit() else 100
            print(f"=== Capture Mode ===")
            capture_new_user(user_name=name, num_samples=samples)
        elif choice == '2':
            print(f"=== Training Mode ===")
            train_model()
        elif choice == '3':
            print(f"=== Recognize Mode ===")
            recognize_users()
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

def main():
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="CNN Face Recognition & Liveness System CLI")
        subparsers = parser.add_subparsers(dest='command', help='Commands')

        # capture command
        capture_parser = subparsers.add_parser('capture', help='Capture face image samples for a user')
        capture_parser.add_argument('--name', type=str, required=True, help='Name of the user')
        capture_parser.add_argument('--samples', type=int, default=100, help='Number of image samples to capture')

        # train command
        train_parser = subparsers.add_parser('train', help='Train the model (Embeddings) based on saved data')

        # recognize command
        recognize_parser = subparsers.add_parser('recognize', help='Start live camera for recognition and liveness detection')

        args = parser.parse_args()

        if args.command == 'capture':
            print(f"=== Capture Mode ===")
            capture_new_user(user_name=args.name, num_samples=args.samples)
        
        elif args.command == 'train':
            print(f"=== Training Mode ===")
            train_model()

        elif args.command == 'recognize':
            print(f"=== Recognize Mode ===")
            recognize_users()
        
        else:
            parser.print_help()
            sys.exit(1)
    else:
        interactive_mode()

if __name__ == '__main__':
    main()
