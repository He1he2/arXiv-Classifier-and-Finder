import json
import pandas as pd
import numpy as np
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
nltk.download('punkt_tab')
nltk.download('stopwords')

category_map = {
'acc-phys': 'Accelerator Physics',
'adap-org': 'Not available',
'q-bio': 'Not available',
'cond-mat': 'Not available',
'chao-dyn': 'Not available',
'patt-sol': 'Not available',
'dg-ga': 'Not available',
'solv-int': 'Not available',
'bayes-an': 'Not available',
'comp-gas': 'Not available',
'alg-geom': 'Not available',
'funct-an': 'Not available',
'q-alg': 'Not available',
'ao-sci': 'Not available',
'atom-ph': 'Atomic Physics',
'chem-ph': 'Chemical Physics',
'plasm-ph': 'Plasma Physics',
'mtrl-th': 'Not available',
'cmp-lg': 'Not available',
'supr-con': 'Not available',

'econ.GN': 'General Economics',
'econ.TH': 'Theoretical Economics',
'eess.SY': 'Systems and Control',

'astro-ph': 'Astrophysics',
'astro-ph.CO': 'Cosmology and Nongalactic Astrophysics',
'astro-ph.EP': 'Earth and Planetary Astrophysics',
'astro-ph.GA': 'Astrophysics of Galaxies',
'astro-ph.HE': 'High Energy Astrophysical Phenomena',
'astro-ph.IM': 'Instrumentation and Methods for Astrophysics',
'astro-ph.SR': 'Solar and Stellar Astrophysics',
'cond-mat.dis-nn': 'Disordered Systems and Neural Networks',
'cond-mat.mes-hall': 'Mesoscale and Nanoscale Physics',
'cond-mat.mtrl-sci': 'Materials Science',
'cond-mat.other': 'Other Condensed Matter',
'cond-mat.quant-gas': 'Quantum Gases',
'cond-mat.soft': 'Soft Condensed Matter',
'cond-mat.stat-mech': 'Statistical Mechanics',
'cond-mat.str-el': 'Strongly Correlated Electrons',
'cond-mat.supr-con': 'Superconductivity',
'cs.AI': 'Artificial Intelligence',
'cs.AR': 'Hardware Architecture',
'cs.CC': 'Computational Complexity',
'cs.CE': 'Computational Engineering, Finance, and Science',
'cs.CG': 'Computational Geometry',
'cs.CL': 'Computation and Language',
'cs.CR': 'Cryptography and Security',
'cs.CV': 'Computer Vision and Pattern Recognition',
'cs.CY': 'Computers and Society',
'cs.DB': 'Databases',
'cs.DC': 'Distributed, Parallel, and Cluster Computing',
'cs.DL': 'Digital Libraries',
'cs.DM': 'Discrete Mathematics',
'cs.DS': 'Data Structures and Algorithms',
'cs.ET': 'Emerging Technologies',
'cs.FL': 'Formal Languages and Automata Theory',
'cs.GL': 'General Literature',
'cs.GR': 'Graphics',
'cs.GT': 'Computer Science and Game Theory',
'cs.HC': 'Human-Computer Interaction',
'cs.IR': 'Information Retrieval',
'cs.IT': 'Information Theory',
'cs.LG': 'Machine Learning',
'cs.LO': 'Logic in Computer Science',
'cs.MA': 'Multiagent Systems',
'cs.MM': 'Multimedia',
'cs.MS': 'Mathematical Software',
'cs.NA': 'Numerical Analysis',
'cs.NE': 'Neural and Evolutionary Computing',
'cs.NI': 'Networking and Internet Architecture',
'cs.OH': 'Other Computer Science',
'cs.OS': 'Operating Systems',
'cs.PF': 'Performance',
'cs.PL': 'Programming Languages',
'cs.RO': 'Robotics',
'cs.SC': 'Symbolic Computation',
'cs.SD': 'Sound',
'cs.SE': 'Software Engineering',
'cs.SI': 'Social and Information Networks',
'cs.SY': 'Systems and Control',
'econ.EM': 'Econometrics',
'eess.AS': 'Audio and Speech Processing',
'eess.IV': 'Image and Video Processing',
'eess.SP': 'Signal Processing',
'gr-qc': 'General Relativity and Quantum Cosmology',
'hep-ex': 'High Energy Physics - Experiment',
'hep-lat': 'High Energy Physics - Lattice',
'hep-ph': 'High Energy Physics - Phenomenology',
'hep-th': 'High Energy Physics - Theory',
'math.AC': 'Commutative Algebra',
'math.AG': 'Algebraic Geometry',
'math.AP': 'Analysis of PDEs',
'math.AT': 'Algebraic Topology',
'math.CA': 'Classical Analysis and ODEs',
'math.CO': 'Combinatorics',
'math.CT': 'Category Theory',
'math.CV': 'Complex Variables',
'math.DG': 'Differential Geometry',
'math.DS': 'Dynamical Systems',
'math.FA': 'Functional Analysis',
'math.GM': 'General Mathematics',
'math.GN': 'General Topology',
'math.GR': 'Group Theory',
'math.GT': 'Geometric Topology',
'math.HO': 'History and Overview',
'math.IT': 'Information Theory',
'math.KT': 'K-Theory and Homology',
'math.LO': 'Logic',
'math.MG': 'Metric Geometry',
'math.MP': 'Mathematical Physics',
'math.NA': 'Numerical Analysis',
'math.NT': 'Number Theory',
'math.OA': 'Operator Algebras',
'math.OC': 'Optimization and Control',
'math.PR': 'Probability',
'math.QA': 'Quantum Algebra',
'math.RA': 'Rings and Algebras',
'math.RT': 'Representation Theory',
'math.SG': 'Symplectic Geometry',
'math.SP': 'Spectral Theory',
'math.ST': 'Statistics Theory',
'math-ph': 'Mathematical Physics',
'nlin.AO': 'Adaptation and Self-Organizing Systems',
'nlin.CD': 'Chaotic Dynamics',
'nlin.CG': 'Cellular Automata and Lattice Gases',
'nlin.PS': 'Pattern Formation and Solitons',
'nlin.SI': 'Exactly Solvable and Integrable Systems',
'nucl-ex': 'Nuclear Experiment',
'nucl-th': 'Nuclear Theory',
'physics.acc-ph': 'Accelerator Physics',
'physics.ao-ph': 'Atmospheric and Oceanic Physics',
'physics.app-ph': 'Applied Physics',
'physics.atm-clus': 'Atomic and Molecular Clusters',
'physics.atom-ph': 'Atomic Physics',
'physics.bio-ph': 'Biological Physics',
'physics.chem-ph': 'Chemical Physics',
'physics.class-ph': 'Classical Physics',
'physics.comp-ph': 'Computational Physics',
'physics.data-an': 'Data Analysis, Statistics and Probability',
'physics.ed-ph': 'Physics Education',
'physics.flu-dyn': 'Fluid Dynamics',
'physics.gen-ph': 'General Physics',
'physics.geo-ph': 'Geophysics',
'physics.hist-ph': 'History and Philosophy of Physics',
'physics.ins-det': 'Instrumentation and Detectors',
'physics.med-ph': 'Medical Physics',
'physics.optics': 'Optics',
'physics.plasm-ph': 'Plasma Physics',
'physics.pop-ph': 'Popular Physics',
'physics.soc-ph': 'Physics and Society',
'physics.space-ph': 'Space Physics',
'q-bio.BM': 'Biomolecules',
'q-bio.CB': 'Cell Behavior',
'q-bio.GN': 'Genomics',
'q-bio.MN': 'Molecular Networks',
'q-bio.NC': 'Neurons and Cognition',
'q-bio.OT': 'Other Quantitative Biology',
'q-bio.PE': 'Populations and Evolution',
'q-bio.QM': 'Quantitative Methods',
'q-bio.SC': 'Subcellular Processes',
'q-bio.TO': 'Tissues and Organs',
'q-fin.CP': 'Computational Finance',
'q-fin.EC': 'Economics',
'q-fin.GN': 'General Finance',
'q-fin.MF': 'Mathematical Finance',
'q-fin.PM': 'Portfolio Management',
'q-fin.PR': 'Pricing of Securities',
'q-fin.RM': 'Risk Management',
'q-fin.ST': 'Statistical Finance',
'q-fin.TR': 'Trading and Market Microstructure',
'quant-ph': 'Quantum Physics',
'stat.AP': 'Applications',
'stat.CO': 'Computation',
'stat.ME': 'Methodology',
'stat.ML': 'Machine Learning',
'stat.OT': 'Other Statistics',
'stat.TH': 'Statistics Theory'
}

# 加载数据集，截取对应时间的数据
def load_filtered_data(filepath, start_year, end_year):
    filtered_data = []
    with open(filepath, 'r') as f:
        for line in f:
            paper = json.loads(line)
            update_date = paper.get("update_date")
            if update_date:
                paper_date = datetime.strptime(update_date, "%Y-%m-%d")
                if start_year <= paper_date.year <= end_year:
                    # Extract only the required columns
                    filtered_data.append({
                        "id": paper.get("id"),
                        "title": paper.get("title"),
                        "abstract": paper.get("abstract"),
                        "categories": paper.get("categories"),
                    })
    
    return filtered_data



def get_cat_text(x):
    cat_text = ''
    cat_list = x.split(' ')
    for i, item in enumerate(cat_list):

        cat_name = category_map[item]
        if cat_name != 'Not available':
            if i == 0:
                cat_text = cat_name
            else:
                cat_text = cat_text + ', ' + cat_name
    cat_text = cat_text.strip()

    return cat_text



# 删除换行符
def clean_text(x):
    new_text = x.replace("\n", " ")
    new_text = new_text.strip()

    return new_text



def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)




if __name__ == '__main__':

    # file_path ="./arxiv-metadata-oai-snapshot.json"

    # start_year = 2023
    # end_year = 2024

    # # 加载23~24年数据
    # filtered_data = load_filtered_data(file_path, start_year, end_year) 

    # # 转换成DataFrame格式
    # df = pd.DataFrame(filtered_data)

    # print("截取23~24年数据，共{}篇论文".format(len(df)))

    # df['cat_text'] = df['categories'].apply(get_cat_text)

    # print("转换特征")
    # df['title'] = df['title'].apply(clean_text)
    # df['abstract'] = df['abstract'].apply(clean_text)

    # print("删除换行符")

    # stop_words = set(stopwords.words('english'))

    # df['processed_abstract'] = df['abstract'].apply(preprocess_text)

    # print("删除停用词")

    # df['text'] = df['title'] + ' {title} ' + df['processed_abstract']

    # print("合并标题和摘要")

    # df.to_json("processed_data.json", orient="records", lines=True, force_ascii=False)

    # print("数据保存成功")

    
    df = pd.read_json("processed_data.json", orient="records", lines=True)
    print(df.head())
    chunk_list = list(df['text'])
    arxiv_id_list = list(df['id'])
    cat_list = list(df['cat_text'])

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("./models/all-MiniLM-L6-v2")
    print("模型加载成功")
    embeddings = model.encode(chunk_list)

    np.savez_compressed('compressed_array.npz', array_data=embeddings)

    print("embeddings 创建完成")