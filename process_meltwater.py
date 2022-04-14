import pandas as pd
import re

"""
TO DO: Clean up the meltwater url and then do match() instead of search(), or some more rigorous search,
because someotimes different domains can be subsets of each other. for example: wausaudailyherald.com and dailyherald.com
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

outlets = pd.read_csv("news sources.csv")
meltwater = pd.read_csv("meltwater.csv")

outlets.dropna(subset=["Website"],inplace=True)
outlets["Website"] = outlets["Website"].str.lower()


# for i in range(80,90):
#     print(outlets["Website"].iloc[i])
#     print(match(outlets["Website"].iloc[i]))
#     print("-------")

outlets["domain"] = outlets["Website"].apply(clean)

meltwater["Publication2"] = meltwater["URL"].apply(find_dom)

meltwater.to_csv("output.csv")