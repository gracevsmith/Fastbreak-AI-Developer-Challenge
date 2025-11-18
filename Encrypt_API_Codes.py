
## import packages
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import getpass
import json
import os


def derive_key(password: str, salt: bytes) -> bytes:
    """Derive a secure key from password"""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))

def encrypt_secrets():
    """Encrypt secrets with a password and save to file"""
    
    ## Get password from user
    password = getpass.getpass("Enter encryption password: ")
    confirm_password = getpass.getpass("Confirm password: ")
    
    if password != confirm_password:
        print("Passwords don't match!")
        return
    
    ## Create salt (this can be stored publicly)
    salt = os.urandom(16)
    
    ## Derive key from password
    key = derive_key(password, salt)
    cipher = Fernet(key)
    
    ## Get secrets from user
    secrets = {}
    print("\nEnter your secrets (press Enter with empty value to finish):")
    while True:
        key_name = input("Secret name (e.g., OPENAI_API_KEY): ").strip()
        if not key_name:
            break
        secret_value = getpass.getpass(f"Value for {key_name}: ")
        secrets[key_name] = secret_value
    
    if not secrets:
        print("No secrets provided!")
        return
    
    # Encrypt the secrets
    secrets_json = json.dumps(secrets)
    encrypted_data = cipher.encrypt(secrets_json.encode())
    
    # Save encrypted file and salt
    with open('secrets.encrypted', 'wb') as f:
        f.write(encrypted_data)
    
    with open('salt.bin', 'wb') as f:
        f.write(salt)
    
    print(f" Successfully encrypted {len(secrets)} secrets!")

def decrypt_secrets():
    """
    Decrypt secrets using password
    """
    
    if not os.path.exists('secrets.encrypted') or not os.path.exists('salt.bin'):
        print("Encrypted files not found!")
        return
    
    password = getpass.getpass("Enter decryption password: ")
    
    # Read salt
    with open('salt.bin', 'rb') as f:
        salt = f.read()
    
    # Derive key
    key = derive_key(password, salt)
    cipher = Fernet(key)
    
    # Read and decrypt
    with open('secrets.encrypted', 'rb') as f:
        encrypted_data = f.read()
    
    try:
        decrypted_data = cipher.decrypt(encrypted_data)
        secrets = json.loads(decrypted_data.decode())
        print("Sucessfully decrypted API codes")
        return secrets
        
    except Exception as e:
        print("Decryption failed: Wrong password or corrupted file")
        return None

if __name__ == "__main__":
    print("1. Encrypt new secrets")
    print("2. Decrypt existing secrets")
    choice = input("Choose (1/2): ")
    
    if choice == "1":
        encrypt_secrets()
    elif choice == "2":
        secrets = decrypt_secrets()
        if secrets:
            print("\nDecrypted secrets:")
            for key, value in secrets.items():
                print(f"{key}: {'*' * len(value)}")
    else:
        print("Invalid choice")