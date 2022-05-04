import pandas as pd
import re
from bs4 import BeautifulSoup
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

"""
TO DO: 

A. Clean up the meltwater url and then do match() instead of search(), or some more rigorous search,
because someotimes different domains can be subsets of each other. for example: wausaudailyherald.com and dailyherald.com.

B. Sometimes the domains also seem to extend beyong the first forward slash, so trimming that always is not a good idea. For example:
https://www.hngnews.com/lake_mills_leader/ and https://www.hngnews.com/lodi_enterprise/ are different.

C. read_slug should check if a pickled output already exists.
"""


head = re.compile("(https://www.|https://|http://www.|http://|www.)")
tailslash = re.compile("/$")

def clean(s):
    return tailslash.sub("",(head.sub("",s)))

def find_dom(s):
    for i in range(len(outlets)):
        domain = outlets["domain"].iloc[i]
        #print(domain)
        if re.search(domain,s) is None:
            pass
        else:
            return outlets["Publication"].iloc[i]

def extract_alltext(URL):
    """
    This func is not acutally being used because it extracts **all** the text, picking up things on the webpage that are not the article.
    There is room for improvement.
    """
    r = requests.get(URL).text
    soup = BeautifulSoup(r,"lxml")
    return soup.get_text()

def extract_article(URL):
    """
    Better than the above func extract_alltext, for most cases (see jupyter notebook for an exception)
    """
    r = requests.get(URL).text
    soup = BeautifulSoup(r,"lxml")
    out = ''
    for line in soup.findAll("p"):
        if line.string == None:
            pass
        else:
            out += line.string
    return out


def assign_outlet():

    outlets = pd.read_csv("news sources.csv")
    meltwater = pd.read_csv("meltwater.csv")

    outlets.dropna(subset=["Website"],inplace=True)
    outlets["Website"] = outlets["Website"].str.lower()


    # for i in range(80,90):
    #     print(outlets["Website"].iloc[i])
    #     print(match(outlets["Website"].iloc[i]))
    #     print("-------")

    outlets["domain"] = outlets["Website"].apply(clean)

    meltwater["Pub_SuggestedName"] = meltwater["URL"].apply(find_dom)

    meltwater.to_csv("output.csv")


def read_slugs():

    """
    This function reads existing slugs; i.e., it accesses WCIJ stories correspoding to slug name,
    and pickles the result. Below I restrict the time range because the full slugs list is quite long.
    """

    #I downloaded from the Google sheets the "Slug Info sheet of the 2021 tracker"
    slugs = pd.read_csv("slug_info.csv", parse_dates=[1])

    #time cutoff - otherwise there are too many slugs to check
    timerange = pd.datetime.dt(2021,6,1)
    slugs = slugs[slugs["Date"]>timerange]

    articles = []

    print("Reading existing slugs.....")

    for story in range(len(slugs)):
        print(slugs["Date"].iloc[story].date())
        print(slugs["Slug"].iloc[story])
        print(slugs["Link"].iloc[story])
        
        URL_temp = slugs["Link"].iloc[story]
        articles.append(extract_article(URL_temp))
        print("----------------------------------------------------")

    with open("slugs.pickle","wb") as f:
        pickle.dump(articles, f)


def assign_slug():

    with open("slugs.pickle","rb") as f:
        articles = pickle.load(f)

    meltwater = pd.read_csv("meltwater.csv")
    
    slugs = pd.read_csv("slug_info.csv", parse_dates=[1])
    timerange = pd.datetime.dt(2021,6,1)
    slugs = slugs[slugs["Date"]>timerange]
    
    assigned_slug = []
    confidence = []

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(articles)
 
    print("Assigning slugs.....")

    for i in range(len(meltwater)):

        print(meltwater["URL"].iloc[i])
        try:
            """
            Noting the exception in the output should be helpful,
            because otherwise not specifying the exception is bad practice.
            """
            Y = vectorizer.transform([extract_article(meltwater["URL"].iloc[i])])
            slugs["tmp"]=cosine_similarity(X,Y)
            slugs.sort_values(by="tmp",ascending=False,inplace=True)
            #print(slugs[["Slug","tmp"]].sort_values(by="tmp",ascending=False).head(3))
            assigned_slug.append(slugs["Slug"].iloc[0])
            confidence.append("{:.2f},{:.2f}".format(slugs["tmp"].iloc[0],slugs["tmp"].iloc[1]))

        # except requests.exceptions.ConnectionError:
        #     assigned_slug.append("NA")

        # except requests.exceptions.InvalidSchema:
        #     assigned_slug.append("NA")    
        except Exception as e:
            assigned_slug.append(e)
            confidence.append("NA")        

    meltwater["SLUG"] = assigned_slug
    meltwater["confidence"] = confidence
    meltwater.to_csv("TMP3.csv")



# read_slugs()
assign_slug()



