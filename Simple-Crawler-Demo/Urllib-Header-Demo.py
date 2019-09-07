import urllib.request

global USER_AGENTS
tar_url = "http://tieba.baidu.com/f?kw=%E5%8D%8E%E4%B8%BA&red_tag=z2304212741"
my_request = urllib.request.Request(tar_url)

''' 
Set header for crawler
with the intension to stimulate a web browser and avoid 403 mistake
'''
USER_AGENTS = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36"
my_request.add_header('User-Agent', USER_AGENTS)

'''
Download a targe website
use retrieve function
'''
def dwonloadSite() :
    urllib.request.urlretrieve(url = tar_url, filename = "e:/Programing/MyPyNote/MyPyNotes/Simple-Crawler-Demo/Copy.html")

'''
Try to get data from target site
With a correspond time limitation of 500ms
'''
def timeoutDemo() :
    for i in range(1, 5) :
        try :
            tar_site = urllib.request.urlopen(my_request, timeout = 500).read().decode('utf-8')
            print(len(tar_site))
            break
        except Exception as e :
            print("Errors happen at : " + str(e))

'''
Get Request
The target website read users' request info from the url
Need to pay attention to the encoding
'''
def getReqDemo() :
    tar_url = "https://www.baidu.com/s?wd="
    search_key = urllib.request.quote("爬虫抓取")
    tar_url = tar_url + search_key
    my_request = urllib.request.Request(tar_url)
    print(tar_url)
    timeoutDemo()


'''
POST Request
Let crawler build a form or jsonObject 
to constract the target post request
'''
import urllib.parse
def postReqDemo() :
     tar_url = "https://category.tudou.com/category/c_96.html"
     post_data = urllib.parse.urlencode({
        "YT-ytaccount" : "userAccount",
        "YT-ytpassword" : "794658123"
     }).encode('utf-8')
     my_request = urllib.request.Request(tar_url, post_data)
     timeoutDemo()

'''
Proxy Server
use a remote ip address through vpn 
so ip can be switched when target site banned the 
ip address current using
'''
def usrProxy() :
    my_proxy = urllib.request.ProxyHandler({
        'http' : 'proxy_server_addrs'
    })
    my_opener = urllib.request.build_opener(my_proxy, urllib.request.HTTPHandler)
    urllib.request.install_opener(my_opener)
    timeoutDemo()

'''
Use DebugLog
Open debug log function in different handlers
'''
def openDebugLog() :
    http_handler = urllib.request.HTTPHandler(debuglevel = 1)
    https_handler = urllib.request.HTTPSHandler(debuglevel = 1)
    my_opener = urllib.request.build_opener(http_handler, https_handler)
    urllib.request.install_opener(my_opener)
    timeoutDemo()
     
'''
UrlError
Exception handler in python crawler
Notice :
urllib.error.HTTPError extends urllib.error.URLError 
'''
import urllib.error
def useUrlError() :
    tar_url = "https://blog.csdn.net/hejjunlin/article/details/9902x5722"
    try :
        urllib.request.urlopen(tar_url)
    except urllib.error.URLError as e :
        print(str(e))
        print(e.code)
        print(e.reason)


if __name__ == "__main__":
    useUrlError()

