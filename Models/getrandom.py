import random
import string

def get_random():
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choices(characters, k=10))
    return random_string