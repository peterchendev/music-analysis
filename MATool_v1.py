import librosa
import numpy as np
import matplotlib.pyplot as plt
import scipy
import os



pitch_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

FLAT_TO_SHARP = {
        'Db': 'C#',
        'Eb': 'D#',
        'Gb': 'F#',
        'Ab': 'G#',
        'Bb': 'A#'
    }

SHARP_TO_FLAT = {v: k for k, v in FLAT_TO_SHARP.items()}

major_pitch_names = ['C Major', 'Db Major', 'D Major', 'Eb Major', 'E Major', 'F Major', 'F# Major', 'G Major', 'Ab Major', 'A Major', 'Bb Major', 'B Major']
minor_pitch_names = ['C Minor', 'C# Minor', 'D Minor', 'Eb Minor', 'E Minor', 'F Minor', 'F# Minor', 'G Minor', 'G# Minor', 'A Minor', 'Bb Minor', 'B Minor']
harmonic_pitch_names = ['C Harmonic Minor', 'C# Harmonic Minor', 'D Harmonic Minor', 'Eb Harmonic Minor', 'E Harmonic Minor', 'F Harmonic Minor', 'F# Harmonic Minor', 'G Harmonic Minor', 'G# Harmonic Minor', 'A Harmonic Minor', 'Bb Harmonic Minor', 'B Harmonic Minor']
dorian_pitch_names = ['C Dorian', 'C# Dorian', 'D Dorian', 'Eb Dorian', 'E Dorian', 'F Dorian', 'F# Dorian', 'G Dorian', 'Ab Dorian', 'A Dorian', 'Bb Dorian', 'B Dorian']
phrygian_pitch_names = ['C Phrygian', 'C# Phrygian', 'D Phrygian', 'D# Phrygian', 'E Phrygian', 'F Phrygian', 'F# Phrygian', 'G Phrygian', 'G# Phrygian', 'A Phrygian', 'Bb Phrygian', 'B Phrygian']
locrian_pitch_names = ['C Locrian', 'C# Locrian', 'D Locrian', 'D# Locrian', 'E Locrian', 'F Locrian', 'F# Locrian', 'G Locrian', 'G# Locrian', 'A Locrian', 'Bb Locrian', 'B Locrian']
lydian_pitch_names = ['C Lydian', 'Db Lydian', 'D Lydian', 'Eb Lydian', 'E Lydian', 'F Lydian', 'F# Lydian', 'G Lydian', 'Ab Lydian', 'A Lydian', 'Bb Lydian', 'B Lydian']
mixolydian_pitch_names = ['C Mixolydian', 'C# Mixolydian', 'D Mixolydian', 'Eb Mixolydian', 'E Mixolydian', 'F Mixolydian', 'F# Mixolydian', 'G Mixolydian', 'Ab Mixolydian', 'A Mixolydian', 'Bb Mixolydian', 'B Mixolydian']
phrygian_dominant_pitch_names = ['C Phrygian Dominant', 'C# Phrygian Dominant', 'D Phrygian Dominant', 'D# Phrygian Dominant', 'E Phrygian Dominant', 'F Phrygian Dominant', 'F# Phrygian Dominant', 'G Phrygian Dominant', 'G# Phrygian Dominant', 'A Phrygian Dominant', 'Bb Phrygian Dominant', 'B Phrygian Dominant']   


All_names = major_pitch_names + minor_pitch_names + harmonic_pitch_names + dorian_pitch_names + phrygian_pitch_names + locrian_pitch_names + lydian_pitch_names + mixolydian_pitch_names + phrygian_dominant_pitch_names

enharmonic = {
        'Db': 'C#', 'Eb': 'D#', 'Fb': 'E', 'Gb': 'F#', 
        'Ab': 'G#', 'Bb': 'A#', 'Cb': 'B'
    }


major_scales = {
        'C':  {0, 2, 4, 5, 7, 9, 11},
        'Db': {1, 3, 5, 6, 8, 10, 0},
        'D':  {2, 4, 6, 7, 9, 11, 1},
        'Eb': {3, 5, 7, 8, 10, 0, 2},
        'E':  {4, 6, 8, 9, 11, 1, 3},
        'F':  {5, 7, 9, 10, 0, 2, 4},
        'F#': {6, 8, 10, 11, 1, 3, 5},
        'G':  {7, 9, 11, 0, 2, 4, 6},
        'Ab': {8, 10, 0, 1, 3, 5, 7},
        'A':  {9, 11, 1, 2, 4, 6, 8},
        'Bb': {10, 0, 2, 3, 5, 7, 9},
        'B':  {11, 1, 3, 4, 6, 8, 10},
    }

def load_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)  # Load audio file, sr=None to preserve original sampling rate
        filename = os.path.basename(file_path)
        return y, sr, filename
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None, None, None 

def detect_tempo(y, sr):
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    #Alternative calcuation
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    beat_intervals = np.diff(beat_times)
    bpm_from_intervals = 60 / np.mean(beat_intervals)

    return tempo, beat_frames, bpm_from_intervals

def chroma_features(y, sr):
    chromagram = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chromagram, axis=1)
    return chromagram, chroma_mean

def chromogram_heatmap(chromagram, sr, filename):
    fig = plt.figure()
    librosa.display.specshow(chromagram, sr=sr, x_axis = "time", y_axis = "chroma")
    plt.colorbar()
    plt.title(f"{filename}: Chromagram Heatmap")
    plt.show()

def chromogram_bar(chromagram, chroma_mean, sr, filename):
    chroma_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    plt.figure(figsize=(10, 6))
    plt.bar(chroma_labels, chroma_mean, color='skyblue')
    plt.xlabel('Chroma')
    plt.ylabel('Average Intensity')
    plt.title(f"{filename}: Average Chroma Intensity")
    plt.show()

def music21_key_analysis(y, sr, chroma_mean):
    from music21 import stream, key, note
    from music21.analysis.discrete import KrumhanslSchmuckler
    analyzer = KrumhanslSchmuckler()

    s = stream.Stream()

    for pitch_name, value in zip(pitch_names, chroma_mean):
        i = int(value * 100)
        while i > 0:
            n = note.Note(pitch_name)
            s.append(n)
            i -= 1


    result = analyzer.getSolution(s)
    return result

def parent_key_detection(chroma_mean):

    parent_key_scores = {}
    for key_name, scale_notes in major_scales.items():
        parent_key_scores[key_name] = sum(chroma_mean[i] for i in scale_notes)

    top3_keys = sorted(parent_key_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    return top3_keys

def mode_detection(chroma_mean):

    c_lydian = [8.0, 1.0, 3.0, 1.0, 6.0, 1.0, 3.0, 6.0, 1.0, 3.0, 1.0, 3.0]
    #             C    C#   D    D#   E    F    F#   G    G#   A    A#   B

    c_major = [8.0, 1.0, 3.0, 1.0, 6.0, 3.0, 1.0, 6.0, 1.0, 3.0, 1.0, 3.0]
    #           C    C#   D    D#   E    F    F#   G    G#   A    A#   B

    c_mixolydian = [8.0, 1.0, 3.0, 1.0, 6.0, 3.0, 1.0, 6.0, 1.0, 3.0, 3.0, 1.0]
    #                C    C#   D    D#   E    F    F#   G    G#   A    A#   B

    c_dorian = [8.0, 1.0, 3.0, 6.0, 1.0, 3.0, 1.0, 6.0, 1.0, 3.0, 3.0, 1.0]
    #            C    C#   D    D#   E    F    F#   G    G#   A    A#   B

    c_natural_minor = [8.0, 1.0, 3.0, 6.0, 1.0, 3.0, 1.0, 6.0, 3.0, 1.0, 3.0, 1.0]
    #                   C    C#   D    D#   E    F    F#   G    G#   A    A#   B  

    c_harmonic_minor = [8.0, 1.0, 3.0, 6.0, 1.0, 3.0, 1.0, 6.0, 3.0, 1.0, 1.0, 3.0]
    #                   C    C#   D    D#   E    F    F#   G    G#   A    A#   B

    c_phrygian = [8.0, 3.0, 1.0, 6.0, 1.0, 3.0, 1.0, 6.0, 3.0, 1.0, 3.0, 1.0]
    #             C    C#   D    D#   E    F    F#   G    G#   A    A#   B

    c_phrygian_dominant = [8.0, 3.0, 1.0, 1.0, 6.0, 3.0, 1.0, 6.0, 3.0, 1.0, 3.0, 1.0]
    #                       C    C#   D    D#   E    F    F#   G    G#   A    A#   B

    c_locrian = [8.0, 3.0, 1.0, 6.0, 1.0, 3.0, 6.0, 1.0, 3.0, 1.0, 3.0, 1.0]
    #             C    C#   D    D#   E    F    F#   G    G#   A    A#   B

    harmonic_minor_profiles = []
    for i in range(12):
        harmonic_minor_profiles.append(np.roll(c_harmonic_minor, i))

    major_profiles = []
    for i in range(12):
        major_profiles.append(np.roll(c_major, i))

    minor_profiles = []
    for i in range(12):
        minor_profiles.append(np.roll(c_natural_minor, i))

    dorian_profiles = []
    for i in range(12):
        dorian_profiles.append(np.roll(c_dorian, i))

    phrygian_profiles = []
    for i in range(12):
        phrygian_profiles.append(np.roll(c_phrygian, i))

    locrian_profiles = []
    for i in range(12):
        locrian_profiles.append(np.roll(c_locrian, i))

    lydian_profiles = []
    for i in range(12):
        lydian_profiles.append(np.roll(c_lydian, i))

    mixolydian_profiles = []
    for i in range(12):
        mixolydian_profiles.append(np.roll(c_mixolydian, i))

    phrygian_dominant_profiles = []
    for i in range(12):
        phrygian_dominant_profiles.append(np.roll(c_phrygian_dominant, i))


    scores = []
    for profile in major_profiles:
        score = np.corrcoef(chroma_mean, profile)
        scores.append(score[0, 1])
    for profile in minor_profiles:
        score = np.corrcoef(chroma_mean, profile)
        scores.append(score[0, 1])
    for profile in harmonic_minor_profiles:
        score = np.corrcoef(chroma_mean, profile)
        scores.append(score[0, 1])
    for profile in dorian_profiles:
        score = np.corrcoef(chroma_mean, profile)
        scores.append(score[0, 1])
    for profile in phrygian_profiles:
        score = np.corrcoef(chroma_mean, profile)
        scores.append(score[0, 1])
    for profile in locrian_profiles:
        score = np.corrcoef(chroma_mean, profile)
        scores.append(score[0, 1])
    for profile in lydian_profiles:
        score = np.corrcoef(chroma_mean, profile)
        scores.append(score[0, 1])
    for profile in mixolydian_profiles:
        score = np.corrcoef(chroma_mean, profile)
        scores.append(score[0, 1])
    for profile in phrygian_dominant_profiles:
        score = np.corrcoef(chroma_mean, profile)
        scores.append(score[0, 1])

    top3_indices = np.argsort(scores)[-3:][::-1]
    return top3_indices, scores

def get_parent_key(mode_result):
        
    mode_to_parent_offset = {
        'Major': 0,
        'Dorian': 10,
        'Phrygian': 8,
        'Phrygian Dominant': 8,
        'Lydian': 7,
        'Mixolydian': 5,
        'Minor': 3,
        'Harmonic Minor': 3,
        'Locrian': 1
    }

    # Split "D Mixolydian" into tonic and mode type
    parts = mode_result.split(' ', 1)
    tonic = parts[0]
    mode_type = parts[1]
    
    if tonic in FLAT_TO_SHARP:
        tonic = FLAT_TO_SHARP[tonic]

    # Get tonic index
    tonic_index = pitch_names.index(tonic)
    
    # Get offset
    offset = mode_to_parent_offset.get(mode_type, 0)
    
    # Calculate parent key index
    parent_index = (tonic_index + offset) % 12
    
    return pitch_names[parent_index]

def notes_are_equivalent(note1, note2):
        """Returns True if notes sound identical (e.g., A# and Bb)."""
        n1_sharp = FLAT_TO_SHARP.get(note1, note1)
        n2_sharp = FLAT_TO_SHARP.get(note2, note2)
        return n1_sharp == n2_sharp

def Coherence_Check(top3_indices, top3_keys):
    

    mode_result = top3_indices


    print("=== Modal Alignment & Coherence Check ===")
    print("Verifying that predicted modes resolve to the detected parent key signatures:\n")
    print("Cross Validation:")
    for i in top3_indices:
        mode_result = All_names[i]
        parent = get_parent_key(mode_result) # e.g., returns "A#"
        
        # Check if ANY key in top3_keys is an enharmonic match for our parent note
        is_match = False
        for k, _ in top3_keys:
            if notes_are_equivalent(parent, k):
                is_match = True
                break
                
        match = "✅" if is_match else "❌"
        
        # Clean up output string so humans see 'Bb' instead of 'A#' if the target uses flats
        # We grab the first key in top3_keys to check if the dataset prefers flats ('b')
        sample_key = top3_keys[0][0] if top3_keys else ""
        if "b" in sample_key:
            display_parent = SHARP_TO_FLAT.get(parent, parent)
        else:
            display_parent = parent

        print(f"  {mode_result} → Parent Key: {display_parent} {match}")

def summary_report(filename, tempo, bpm_from_intervals, result, top3_keys, top3_indices, scores):

    ##10. Summary Cell
    print(f"File: {filename}")

    print("=" * 40)
    print("AUDIO ANALYSIS SUMMARY")
    print("=" * 40)
    print(f"Estimated Tempo: {tempo} BPM")
    print(f"Tempo from intervals: {bpm_from_intervals:.2f} BPM")
    print()
    print(f"Krumhansl-Schmuckler Key: {result}")
    for alt in result.alternateInterpretations[:4]:
        print(f"  Alternative: {alt}")
    print()
    print("Custom Mode Estimation:")
    for i in top3_indices:
        print(f"{All_names[i]}: {scores[i]:.4f}")
    print("=" * 40)

def confidence_score(top3_indices, top3_keys, scores):
    ##11: Confidence Cell

#Note: uses get_parent_key which is defined elsewhere in the notebook to determine the parent key of the mode


    top_score = scores[top3_indices[0]]
    top_mode = All_names[top3_indices[0]]
    top_parent = get_parent_key(top_mode)

    top_parent_keys = [k for k, _ in top3_keys]

    def normalize_key(key):
        return enharmonic.get(key, key)


    if top_score > 0.7:
        print(f"Confidence: HIGH ({top_score:.4f})")
        print(f"Mode: {top_mode}")
        print(f"Parent Key: {top_parent}")
    elif top_score > 0.5:
        print(f"Confidence: AMBIGUOUS ({top_score:.4f})")
        print(f"Most likely parent key: {top_parent}")
        print("Possible modes (cross validated):")
        for i in top3_indices:
            mode = All_names[i]
            parent = get_parent_key(mode)

            if normalize_key(parent) in [normalize_key(k) for k in top_parent_keys]:
                print(f"  {mode} → Parent Key: {parent}")
    else:
        print(f"Confidence: LOW ({top_score:.4f})")
        print("Tonal center unclear — parent key estimate only:")
        for key, score in top3_keys:
            print(f"  {key}: {score:.4f}")

#Generated Using Copilot for Easier Acess to Information outside of Terminal      
def generate_html_report(filename, tempo, bpm_from_intervals, result,
                     top3_keys, top3_indices, scores,
                     chromagram, chroma_mean, sr,
                     All_names, get_parent_key, notes_are_equivalent,
                     enharmonic, SHARP_TO_FLAT):
    #Generated by COPILOT: may require adjustments compared to terminal output.
    import base64
    import io
    import os
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend so plots save without display
    import matplotlib.pyplot as plt
    import librosa
    import librosa.display
    import numpy as np
    
    def save_plot_as_base64(fig):
        """Save a matplotlib figure as a base64 encoded PNG string."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64
    
    def generate_chromagram_heatmap_b64(chromagram, sr, filename):
        fig, ax = plt.subplots(figsize=(12, 4))
        librosa.display.specshow(chromagram, sr=sr, x_axis='time', y_axis='chroma', ax=ax)
        fig.colorbar(ax.collections[0], ax=ax)
        ax.set_title(f"{filename}: Chromagram Heatmap")
        return save_plot_as_base64(fig)
    
    def generate_chromagram_bar_b64(chroma_mean, filename):
        chroma_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(chroma_labels, chroma_mean, color='#4A9EBF')
        ax.set_xlabel('Pitch Class')
        ax.set_ylabel('Mean Intensity')
        ax.set_title(f"{filename}: Mean Chroma Intensity")
        return save_plot_as_base64(fig)
    
    # ── Confidence logic ──────────────────────────────────────────────────────
    top_score   = scores[top3_indices[0]]
    top_mode    = All_names[top3_indices[0]]
    top_parent  = get_parent_key(top_mode)
    top_parent_keys = [k for k, _ in top3_keys]

    def normalize_key(k):
        return enharmonic.get(k, k)

    if top_score > 0.7:
        confidence_level = "HIGH"
        confidence_color = "#2ecc71"
        confidence_html = f"""
            <p><strong>Mode:</strong> {top_mode}</p>
            <p><strong>Parent Key:</strong> {top_parent}</p>
        """
    elif top_score > 0.5:
        confidence_level = "AMBIGUOUS"
        confidence_color = "#f39c12"
        validated_modes = []
        for i in top3_indices:
            mode = All_names[i]
            parent = get_parent_key(mode)
            if normalize_key(parent) in [normalize_key(k) for k in top_parent_keys]:
                display_parent = SHARP_TO_FLAT.get(parent, parent)
                validated_modes.append(f"<li>{mode} → Parent Key: {display_parent}</li>")
        confidence_html = f"""
            <p><strong>Most likely parent key:</strong> {top_parent}</p>
            <p><strong>Possible modes (cross validated):</strong></p>
            <ul>{''.join(validated_modes)}</ul>
        """
    else:
        confidence_level = "LOW"
        confidence_color = "#e74c3c"
        key_rows = ''.join(
            f"<tr><td>{k}</td><td>{s:.4f}</td></tr>" for k, s in top3_keys
        )
        confidence_html = f"""
            <p>Tonal center unclear — parent key estimate only:</p>
            <table>
                <tr><th>Key</th><th>Score</th></tr>
                {key_rows}
            </table>
        """

    # ── K-S alternatives ──────────────────────────────────────────────────────
    ks_alts = ''.join(
        f"<li>{alt}</li>" for alt in result.alternateInterpretations[:4]
    )

    # ── Custom mode top 3 ─────────────────────────────────────────────────────
    mode_rows = ''.join(
        f"<tr><td>{All_names[i]}</td><td>{scores[i]:.4f}</td></tr>"
        for i in top3_indices
    )

    # ── Parent key top 3 ──────────────────────────────────────────────────────
    parent_rows = ''.join(
        f"<tr><td>{k}</td><td>{s:.4f}</td></tr>" for k, s in top3_keys
    )

    # ── Coherence check ───────────────────────────────────────────────────────
    coherence_rows = []
    for i in top3_indices:
        mode_result = All_names[i]
        parent = get_parent_key(mode_result)
        is_match = any(notes_are_equivalent(parent, k) for k, _ in top3_keys)
        sample_key = top3_keys[0][0] if top3_keys else ""
        display_parent = SHARP_TO_FLAT.get(parent, parent) if "b" in sample_key else parent
        icon = "✅" if is_match else "❌"
        coherence_rows.append(
            f"<tr><td>{mode_result}</td><td>{display_parent}</td><td>{icon}</td></tr>"
        )

    # ── Plots ─────────────────────────────────────────────────────────────────
    heatmap_b64 = generate_chromagram_heatmap_b64(chromagram, sr, filename)
    bar_b64     = generate_chromagram_bar_b64(chroma_mean, filename)

    # ── Assemble HTML ─────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Music Analysis — {filename}</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap" rel="stylesheet">
<style>
:root {{
    --bg:       #0f0f11;
    --surface:  #1a1a1f;
    --border:   #2a2a32;
    --accent:   #7EB8D4;
    --accent2:  #C47EB8;
    --text:     #e8e8ec;
    --muted:    #6b6b78;
    --high:     #2ecc71;
    --amb:      #f39c12;
    --low:      #e74c3c;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 15px;
    line-height: 1.6;
    padding: 2rem;
}}
header {{
    border-bottom: 1px solid var(--border);
    padding-bottom: 1.5rem;
    margin-bottom: 2rem;
}}
header h1 {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    color: var(--accent);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}}
header h2 {{
    font-size: 1.6rem;
    font-weight: 300;
    color: var(--text);
}}
.grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.25rem;
    margin-bottom: 1.25rem;
}}
.card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.4rem;
}}
.card.full {{ grid-column: 1 / -1; }}
.card h3 {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted);
    margin-bottom: 1rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
}}
.stat {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 0.35rem 0;
    border-bottom: 1px solid var(--border);
}}
.stat:last-child {{ border-bottom: none; }}
.stat-label {{ color: var(--muted); font-size: 0.88rem; }}
.stat-value {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.95rem; }}
.badge {{
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}}
.badge-high   {{ background: #1a3a2a; color: var(--high);  border: 1px solid var(--high); }}
.badge-amb    {{ background: #3a2e10; color: var(--amb);   border: 1px solid var(--amb); }}
.badge-low    {{ background: #3a1a1a; color: var(--low);   border: 1px solid var(--low); }}
table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
    margin-top: 0.5rem;
}}
th {{
    text-align: left;
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0.4rem 0.6rem;
    border-bottom: 1px solid var(--border);
}}
td {{
    padding: 0.45rem 0.6rem;
    border-bottom: 1px solid var(--border);
    font-family: 'IBM Plex Mono', monospace;
}}
tr:last-child td {{ border-bottom: none; }}
ul {{ padding-left: 1.2rem; margin-top: 0.4rem; }}
li {{ padding: 0.2rem 0; font-family: 'IBM Plex Mono', monospace; font-size: 0.9rem; }}
img {{ width: 100%; border-radius: 6px; margin-top: 0.5rem; }}
.confidence-detail {{ margin-top: 0.75rem; }}
.confidence-detail p {{ margin-bottom: 0.4rem; font-size: 0.9rem; }}
</style>
</head>
<body>

<header>
<h1>Music Analysis Report</h1>
<h2>{filename}</h2>
</header>

<div class="grid">

<!-- Tempo -->
<div class="card">
    <h3>Tempo</h3>
    <div class="stat">
    <span class="stat-label">Beat Track</span>
    <span class="stat-value">{float(tempo):.2f} BPM</span>
    </div>
    <div class="stat">
    <span class="stat-label">Interval Method</span>
    <span class="stat-value">{bpm_from_intervals:.2f} BPM</span>
    </div>
</div>

<!-- Confidence -->
<div class="card">
    <h3>Confidence</h3>
    <span class="badge badge-{confidence_level.lower()}">{confidence_level}</span>
    <div class="confidence-detail">
    {confidence_html}
    </div>
</div>

<!-- K-S Key Detection -->
<div class="card">
    <h3>Krumhansl-Schmuckler Key</h3>
    <div class="stat">
    <span class="stat-label">Primary</span>
    <span class="stat-value">{result}</span>
    </div>
    <ul style="margin-top:0.6rem">{ks_alts}</ul>
</div>

<!-- Parent Key -->
<div class="card">
    <h3>Parent Key Detection</h3>
    <table>
    <tr><th>Key</th><th>Score</th></tr>
    {parent_rows}
    </table>
</div>

<!-- Custom Mode Estimation -->
<div class="card">
    <h3>Custom Mode Estimation</h3>
    <table>
    <tr><th>Mode</th><th>Score</th></tr>
    {mode_rows}
    </table>
</div>

<!-- Coherence Check -->
<div class="card">
    <h3>Modal Coherence Check</h3>
    <table>
    <tr><th>Mode</th><th>Parent Key</th><th>Match</th></tr>
    {''.join(coherence_rows)}
    </table>
</div>

<!-- Chromagram Heatmap -->
<div class="card full">
    <h3>Chromagram Heatmap</h3>
    <img src="data:image/png;base64,{heatmap_b64}" alt="Chromagram Heatmap">
</div>

<!-- Chromagram Bar -->
<div class="card full">
    <h3>Mean Chroma Intensity</h3>
    <img src="data:image/png;base64,{bar_b64}" alt="Chroma Bar Chart">
</div>

</div>
</body>
</html>"""

    # Save file
    base_name = os.path.splitext(filename)[0]
    report_path = f"{base_name}_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Report saved: {report_path}")
    return report_path


def main():

    while True:
        file_path = input("Enter the path to your audio file: ").strip()
        
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            if not file_path.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                raise ValueError("Unsupported file format. Please use .wav, .mp3, .flac, or .ogg")
            
            print(f"Successfully loaded: {os.path.basename(file_path)}\n")
            break
            
        except FileNotFoundError as e:
            print(f"Error: {e}. Please try again.\n")
        except ValueError as e:
            print(f"Error: {e}. Please try again.\n")
        except Exception as e:
            print(f"Unexpected error: {e}. Please try again.\n")

    
    y, sr, filename = load_audio(file_path)
    tempo, beat_frames, bpm_from_intervals = detect_tempo(y, sr)
    chromagram, chroma_mean = chroma_features(y, sr)
    
    result = music21_key_analysis(y, sr, chroma_mean)
    top3_keys = parent_key_detection(chroma_mean)
    top3_indices, scores = mode_detection(chroma_mean)
    summary_report(filename, tempo, bpm_from_intervals, result, top3_keys, top3_indices, scores)


    #chromogram_heatmap(chromagram, sr, filename)
    #chromogram_bar(chromagram, chroma_mean, sr, filename)


    confidence_score(top3_indices, top3_keys, scores)
    print("\n")
    Coherence_Check(top3_indices, top3_keys)


    generate_html_report(filename, tempo, bpm_from_intervals, result,
                     top3_keys, top3_indices, scores,
                     chromagram, chroma_mean, sr,
                     All_names, get_parent_key, notes_are_equivalent,
                     enharmonic, SHARP_TO_FLAT)

    
main()