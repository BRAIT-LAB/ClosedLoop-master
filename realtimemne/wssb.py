def is_valid(permutation):
    # Check all conditions
    conditions = [
        (1, 7),
        (3, 8),
        (2, 8),
        (4, 9),
        (2, 9),
        (1, 10),
        (1, 7),
        (4, 12),
        (4, 9),
        (1, 11)
    ]

    for (a, b) in conditions:
        pos_a = permutation.index(a)
        pos_b = permutation.index(b)
        if abs(pos_a - pos_b) < 3:
            return False

    # Check alternating condition
    for i in range(len(permutation) - 1):
        if (permutation[i] <= 6 and permutation[i + 1] <= 6) or (permutation[i] > 6 and permutation[i + 1] > 6):
            return False

    return True


def find_permutation(current_permutation, remaining_numbers):
    if not remaining_numbers:
        if is_valid(current_permutation):
            return current_permutation
        else:
            return None

    for i, num in enumerate(remaining_numbers):
        new_permutation = current_permutation + [num]
        new_remaining_numbers = remaining_numbers[:i] + remaining_numbers[i + 1:]
        result = find_permutation(new_permutation, new_remaining_numbers)
        if result:
            return result

    return None


numbers = list(range(1, 13))
result = find_permutation([], numbers)

if result:
    print("Found valid permutation:", result)
else:
    print("No valid permutation found")
