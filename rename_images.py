import os

# Aragorn

folder = r"C:\Users\flavi\PycharmProjects\lotr-lookalike\gallery\aragorn"
files = sorted(os.listdir(folder))

idx = 1
for filename in files:
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
        # Skip files already named like aragorn_01.jpg, etc.
        if filename.lower().startswith('aragorn_'):
            continue
        ext = os.path.splitext(filename)[1]
        new_name = f"aragorn_{idx:02d}{ext}"
        src = os.path.join(folder, filename)
        dst = os.path.join(folder, new_name)
        # If destination exists, increment index until it's unique
        while os.path.exists(dst):
            idx += 1
            new_name = f"aragorn_{idx:02d}{ext}"
            dst = os.path.join(folder, new_name)
        os.rename(src, dst)
        idx += 1

print("Renaming Aragorn completed.")


# Arwen

folder = r"C:\Users\flavi\PycharmProjects\lotr-lookalike\gallery\arwen"
files = sorted(os.listdir(folder))

idx = 1
for filename in files:
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
        # Skip files already named like arwen_01.jpg, etc.
        if filename.lower().startswith('arwen_'):
            continue
        ext = os.path.splitext(filename)[1]
        new_name = f"arwen_{idx:02d}{ext}"
        src = os.path.join(folder, filename)
        dst = os.path.join(folder, new_name)
        # If destination exists, increment index until it's unique
        while os.path.exists(dst):
            idx += 1
            new_name = f"arwen_{idx:02d}{ext}"
            dst = os.path.join(folder, new_name)
        os.rename(src, dst)
        idx += 1

print("Renaming Arwen completed.")

# Eowyn

folder = r"C:\Users\flavi\PycharmProjects\lotr-lookalike\gallery\eowyn"
files = sorted(os.listdir(folder))

idx = 1
for filename in files:
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
        # Skip files already named like eowyn_01.jpg, etc.
        if filename.lower().startswith('eowyn_'):
            continue
        ext = os.path.splitext(filename)[1]
        new_name = f"eowyn_{idx:02d}{ext}"
        src = os.path.join(folder, filename)
        dst = os.path.join(folder, new_name)
        # If destination exists, increment index until it's unique
        while os.path.exists(dst):
            idx += 1
            new_name = f"eowyn_{idx:02d}{ext}"
            dst = os.path.join(folder, new_name)
        os.rename(src, dst)
        idx += 1

print("Renaming Eowyn completed.")

# Frodo

folder = r"C:\Users\flavi\PycharmProjects\lotr-lookalike\gallery\frodo"
files = sorted(os.listdir(folder))

idx = 1
for filename in files:
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
        # Skip files already named like frodo_01.jpg, etc.
        if filename.lower().startswith('frodo_'):
            continue
        ext = os.path.splitext(filename)[1]
        new_name = f"frodo_{idx:02d}{ext}"
        src = os.path.join(folder, filename)
        dst = os.path.join(folder, new_name)
        # If destination exists, increment index until it's unique
        while os.path.exists(dst):
            idx += 1
            new_name = f"frodo_{idx:02d}{ext}"
            dst = os.path.join(folder, new_name)
        os.rename(src, dst)
        idx += 1

print("Renaming Frodo completed.")

# Galadriel

folder = r"C:\Users\flavi\PycharmProjects\lotr-lookalike\gallery\galadriel"
files = sorted(os.listdir(folder))

idx = 1
for filename in files:
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
        # Skip files already named like galadriel_01.jpg, etc.
        if filename.lower().startswith('galadriel_'):
            continue
        ext = os.path.splitext(filename)[1]
        new_name = f"galadriel_{idx:02d}{ext}"
        src = os.path.join(folder, filename)
        dst = os.path.join(folder, new_name)
        # If destination exists, increment index until it's unique
        while os.path.exists(dst):
            idx += 1
            new_name = f"galadriel_{idx:02d}{ext}"
            dst = os.path.join(folder, new_name)
        os.rename(src, dst)
        idx += 1

print("Renaming Galadriel completed.")


# Gandalf

folder = r"C:\Users\flavi\PycharmProjects\lotr-lookalike\gallery\gandalf"
files = sorted(os.listdir(folder))

idx = 1
for filename in files:
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
        # Skip files already named like gandalf_01.jpg, etc.
        if filename.lower().startswith('gandalf_'):
            continue
        ext = os.path.splitext(filename)[1]
        new_name = f"gandalf_{idx:02d}{ext}"
        src = os.path.join(folder, filename)
        dst = os.path.join(folder, new_name)
        # If destination exists, increment index until it's unique
        while os.path.exists(dst):
            idx += 1
            new_name = f"gandalf_{idx:02d}{ext}"
            dst = os.path.join(folder, new_name)
        os.rename(src, dst)
        idx += 1

print("Renaming Gandalf completed.")


# Gimli

folder = r"C:\Users\flavi\PycharmProjects\lotr-lookalike\gallery\gimli"
files = sorted(os.listdir(folder))

idx = 1
for filename in files:
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
        # Skip files already named like gimli_01.jpg, etc.
        if filename.lower().startswith('gimli_'):
            continue
        ext = os.path.splitext(filename)[1]
        new_name = f"gimli_{idx:02d}{ext}"
        src = os.path.join(folder, filename)
        dst = os.path.join(folder, new_name)
        # If destination exists, increment index until it's unique
        while os.path.exists(dst):
            idx += 1
            new_name = f"gimli_{idx:02d}{ext}"
            dst = os.path.join(folder, new_name)
        os.rename(src, dst)
        idx += 1

print("Renaming Gimli completed.")


# Gollum

folder = r"C:\Users\flavi\PycharmProjects\lotr-lookalike\gallery\gollum"
files = sorted(os.listdir(folder))

idx = 1
for filename in files:
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
        # Skip files already named like gollum_01.jpg, etc.
        if filename.lower().startswith('gollum_'):
            continue
        ext = os.path.splitext(filename)[1]
        new_name = f"gollum_{idx:02d}{ext}"
        src = os.path.join(folder, filename)
        dst = os.path.join(folder, new_name)
        # If destination exists, increment index until it's unique
        while os.path.exists(dst):
            idx += 1
            new_name = f"gollum_{idx:02d}{ext}"
            dst = os.path.join(folder, new_name)
        os.rename(src, dst)
        idx += 1

print("Renaming Gollum completed.")

# Legolas

folder = r"C:\Users\flavi\PycharmProjects\lotr-lookalike\gallery\legolas"
files = sorted(os.listdir(folder))

idx = 1
for filename in files:
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
        # Skip files already named like legolas_01.jpg, etc.
        if filename.lower().startswith('legolas_'):
            continue
        ext = os.path.splitext(filename)[1]
        new_name = f"legolas_{idx:02d}{ext}"
        src = os.path.join(folder, filename)
        dst = os.path.join(folder, new_name)
        # If destination exists, increment index until it's unique
        while os.path.exists(dst):
            idx += 1
            new_name = f"legolas_{idx:02d}{ext}"
            dst = os.path.join(folder, new_name)
        os.rename(src, dst)
        idx += 1

print("Renaming Legolas completed.")

# Samwise

folder = r"C:\Users\flavi\PycharmProjects\lotr-lookalike\gallery\samwise"
files = sorted(os.listdir(folder))

idx = 1
for filename in files:
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
        # Skip files already named like samwise_01.jpg, etc.
        if filename.lower().startswith('samwise_'):
            continue
        ext = os.path.splitext(filename)[1]
        new_name = f"samwise_{idx:02d}{ext}"
        src = os.path.join(folder, filename)
        dst = os.path.join(folder, new_name)
        # If destination exists, increment index until it's unique
        while os.path.exists(dst):
            idx += 1
            new_name = f"samwise_{idx:02d}{ext}"
            dst = os.path.join(folder, new_name)
        os.rename(src, dst)
        idx += 1

print("Renaming Samwise completed.")

