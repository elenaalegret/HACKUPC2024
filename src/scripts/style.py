# Custom styles
styles = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Tenor+Sans&display=swap');

body {
    font-family: 'Tenor Sans', sans-serif;
    color: black;
}
.reportview-container {
    background: #ffffff;
}
.stImage {
    border: 2px solid black;
    margin: 0;
    position: relative;
    z-index: 2;
}
.title {
    font-family: 'Tenor Sans', sans-serif;
    border: 1px solid black;
    padding: 10px;
    font-size: 28px;
    font-weight: bold;
    background-color: white;
    text-align: center;
    margin-bottom: 20px;
}

.subheader {
    font-family: 'Tenor Sans', sans-serif;
    border: 1px solid black;
    font-size: 18px;
    background-color: black;
    color: white;
    text-align: center;
    margin: 20px 0;
}
.grid-container {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    grid-template-rows: repeat(3, 80px);
    gap: 0;
    width: 240px;
}
.grid-item {
    width: 80px;
    height: 80px;
}
.grid-item img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}
.stButton > button {
    background-color: black !important;
    color: white !important;
    border-radius: 0 !important;
    font-family: 'Tenor Sans', sans-serif;
}
[data-testid="stSidebar"][aria-expanded="true"] {
    min-width: 180px;
    max-width: 180px;
    border: 1px solid black;
    padding: 10px;
    font-family: 'Tenor Sans', sans-serif !important;
    font-size: 28px;
    font-weight: bold;
    background-color: white;
}
</style>
"""

# Dictionaries of labels
section_labels = {
    1: 'Woman',
    2: 'Men',
    3: 'Kids'
}

types_labels = {
    0: 'Clothes',
    1: 'Shoes',
    2: 'Perfumery',
    4: 'Home'
}

