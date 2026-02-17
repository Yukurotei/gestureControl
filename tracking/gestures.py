def are_fingers_touching(landmark1, landmark2, threshold, min_threshold=0.0):
    dist = ((landmark1.x - landmark2.x)**2 +
            (landmark1.y - landmark2.y)**2 +
            (landmark1.z - landmark2.z)**2)**0.5
    return min_threshold <= dist < threshold


def is_fist(hand_landmarks, curled_fingers_required=3):
    wrist = hand_landmarks[0]

    fingertips = [
        hand_landmarks[8],   # Index
        hand_landmarks[12],  # Middle
        hand_landmarks[16],  # Ring
        hand_landmarks[20],  # Pinky
    ]

    finger_bases = [
        hand_landmarks[5],   # Index MCP
        hand_landmarks[9],   # Middle MCP
        hand_landmarks[13],  # Ring MCP
        hand_landmarks[17],  # Pinky MCP
    ]

    curled_count = 0
    for tip, base in zip(fingertips, finger_bases):
        tip_dist = ((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)**0.5
        base_dist = ((base.x - wrist.x)**2 + (base.y - wrist.y)**2)**0.5
        if tip_dist < base_dist * 1.1:
            curled_count += 1

    return curled_count >= curled_fingers_required


def finger_distance(landmark1, landmark2):
    return ((landmark1.x - landmark2.x)**2 +
            (landmark1.y - landmark2.y)**2 +
            (landmark1.z - landmark2.z)**2)**0.5
