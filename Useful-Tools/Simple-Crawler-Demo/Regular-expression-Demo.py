import re

def switchContent(contentName) :
    print("\n----------", contentName, "----------")

switchContent("Pattern match")
# use () to create groups
pattern = "^([d-x])a(.*)"
text = "PAP MLORAN BUILDING"
# match the whole sentence with flag IGNORECASE
re_result = re.match(pattern, text, flags=re.IGNORECASE|re.S)
if re_result :
    # use group to show the whole match list
    print(re_result.group())
    # use index to select target element
    print("group(1) :", re_result.group(1), "\tgroup(2) :",re_result.group(2))

switchContent("Pattern search")
pattern = "\scl[a-z]+"
text = "manning clark hall"
# use span to get the range of the result
re_result = re.search(pattern, text)
if re_result:
    print(re_result)
    print("span : ", re_result.span(), "\tstart at :", re_result.start(), "\tend at :", re_result.end())
 
switchContent("Pattern replace")
phone = "2233-666-779 ## comments"
# repacle target patterns with ""
re_result = re.sub(r'#.*$', "", phone)
print(re_result)
re_result = re.sub(r'\D', "", phone)
print(re_result)


switchContent("Pattern find")
find_str = "144133161718819"
# compile the pattern
pattern = re.compile("1*")
re_result = pattern.findall(find_str, 3)
print(re_result)
re_result = re.finditer(pattern, find_str)
for it in re_result :
    print(it)
