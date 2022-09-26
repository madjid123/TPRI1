from nltk import ngrams
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import json
# types of params
typeofclass = 2
lang = "en"
n = 2

en_corpus = """
Despite its widespread lack of familiarity, AI is a technology that is transforming every walk of life. It is a wide-ranging tool that enables people to rethink how we integrate information, analyze data, and use the resulting insights to improve decisionmaking. Our hope through this comprehensive overview is to explain AI to an audience of policymakers, opinion leaders, and interested observers, and demonstrate how AI already is altering the world and raising important questions for society, the economy, and governance.

In this paper, we discuss novel applications in finance, national security, health care, criminal justice, transportation, and smart cities, and address issues such as data access problems, algorithmic bias, AI ethics and transparency, and legal liability for AI decisions. We contrast the regulatory approaches of the U.S. and European Union, and close by making a number of recommendations for getting the most out of AI while still protecting important human values.
"""
ar_corpus = """
قال الرئيس التركي رجب طيب أردوغان إن اليونان ستدفع ثمنا باهظا لدورها في -ما سماها- حياكة المؤامرات على تركيا.
وأكد أردوغان أن بلاده لن تتوانى عن الدفاع عن حقوقها ومصالحها بجميع الوسائل المتاحة، وأن ما وصفها بالحشود العسكرية الأجنبية ذات المظهر الاحتلالي المنتشرة في أنحاء اليونان كافة، خطر على اليونانيين وقد تحولها إلى مستنقع، حسب تعبيره.
وأضاف "في حين نبذل جهودا صادقة من أجل إنهاء الحروب والأزمات والتوترات في العالم، نتابع بدقة سياسات جارتنا اليونان التي تفوح منها رائحة الاستفزاز والتحرش".
وجدد الرئيس التركي التأكيد على أنه لا يمكن لليونان أن تضاهي قوة تركيا لا سياسيا ولا اقتصاديا، مشيرا إلى أن "النية الحقيقية للذين يحرضون السياسيين اليونانيين ضدنا هي إعاقة برنامجنا لبناء تركيا عظيمة وقوية، ولكن هذه لعبة خطيرة بالنسبة للسياسيين اليونانيين والدولة اليونانية والشعب اليوناني ومن يستخدمهم كدمى".
"""
typeofclass = int(
    input("to use word based classification use 1, for character based one use 2: "))
lang = input("enter en for english corpus, ar for the arabic one: ")

if lang == "en":
    corpus = en_corpus
else:
    corpus = ar_corpus

corpus = word_tokenize(corpus)

n = int(input("Enter the number of ngrams: "))
if typeofclass == 2:
    corpus = [c for w in corpus for c in w]
n_grams = ngrams(corpus, n)

ngram_fd = nltk.FreqDist(ngrams(corpus, n))


y = sorted(ngram_fd.values(), reverse=True)
x = [i for i in range(len(y))]

plt.loglog(x, y)
plt.xlabel('rank(f)', fontsize=14, fontweight='bold')
plt.ylabel('frequency(r)', fontsize=14, fontweight='bold')
plt.grid(True)
plt.show()
f = open("classement.txt", "a")
f.write(json.dumps(ngram_fd.most_common()))
f.close()
