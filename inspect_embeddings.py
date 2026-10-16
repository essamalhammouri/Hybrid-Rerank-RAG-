from sentence_transformers import SentenceTransformer, util

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
embedder = SentenceTransformer(EMBED_MODEL)

titles = [
    "اصدار شهادة ثانوية عامة باللغة الانجليزية",
    "اصدار كشف علامات امتحان شهادة الثانوية العامة باللغة الانجليزية",
    "طلب مشروحات / هوية / دفتر عائلة / جواز سفر بدل فاقد",
]

vectors = embedder.encode(titles)

for i in range(len(titles)):
    for j in range(i + 1, len(titles)):
        score = util.cos_sim(vectors[i], vectors[j]).item()
        print(f"{score:.3f}  |  {titles[i][:30]}...  vs  {titles[j][:30]}...")