import hashlib
import re
from lxml import etree
import scrapy
import redis

def encode_md5(value):
    if value:
        hl = hashlib.md5()
        hl.update(value.encode(encoding='utf-8'))
        return hl.hexdigest()
    return ''

class ZghjbSpider(scrapy.Spider):
    name = 'zghjb'
    start_urls = [
        'https://www.mee.gov.cn/ywdt/dfnews/',  # 地方快讯
        'https://www.mee.gov.cn/ywdt/hjywnews/',  # 环境要闻
    ]
    redis_conn = redis.Redis(host="124.221.199.74", port=6001)
    redis_key = "hb_hot_news"

    # def start_requests(self):
    #     for url in self.start_urls:

    def parse(self, response):
        resp_url = response.url
        text = response.body.decode("utf-8")
        max_page = re.findall("countPage = (.*?)//共多少页",text)[0]
        yield scrapy.Request(url=resp_url, dont_filter=True, callback=self.get_detail_url)
        for i in range(1,int(max_page)):
            url = resp_url + f"index_{i}.shtml"
            yield scrapy.Request(url=url, dont_filter=True, callback=self.get_detail_url)
        #     break

    def get_detail_url(self, response):
        resp_url = response.url
        text = response.body.decode("utf-8")
        html = etree.HTML(text)
        href_list = html.xpath("//a[@class='cjcx_biaobnan']/@href")

        for href in href_list:
            if not re.match("^\./",href):
                continue
            if "dfnews" in resp_url:
                url = href.replace("./","https://www.mee.gov.cn/ywdt/dfnews/")
            elif "hjywnews" in resp_url:
                url = href.replace("./","https://www.mee.gov.cn/ywdt/hjywnews/")
            else:
                continue
            yield scrapy.Request(url=url, dont_filter=True, callback=self.get_detail)
            # break

    def get_detail(self, response):
        # print(f"get_detail: {response.url}")
        text = response.body.decode("utf-8")
        html = etree.HTML(text)
        trs = html.xpath("//div[@class='TRS_Editor']")[0]
        # print(html.xpath("//div[@class='TRS_Editor']//text()"))
        htmlContent = etree.tostring(trs,encoding='utf-8',method='html',pretty_print=True).decode('utf-8')
        more_url = re.findall("<a href=\"(.*?)\">.*?更多内容，点击阅读</a>",htmlContent)
        exam_url = re.findall("典型案例.*?<a href=\"(.*?)\">",text)
        # if "更多内容，点击阅读" in htmlContent:
        #     print(htmlContent)
        if more_url:
            for url in more_url:
                if self.redis_conn.sadd(self.redis_key,url):
                    yield scrapy.Request(url=url, dont_filter=True, callback=self.get_more_detail)

        if exam_url:
            for url in exam_url:
                if self.redis_conn.sadd(self.redis_key, url):
                    yield scrapy.Request(url=url, dont_filter=True, callback=self.get_more_detail)

        if not more_url and not exam_url:
            if self.redis_conn.sadd(self.redis_key, response.url):
                yield scrapy.Request(url=response.url, dont_filter=True, callback=self.get_more_detail)

    def get_more_detail(self, response):
        resp_url = response.url
        text = response.body.decode("utf-8")
        html = etree.HTML(text)
        trs = html.xpath("//div[@class='TRS_Editor']")[0]
        htmlContent = etree.tostring(trs, encoding='utf-8', method='html', pretty_print=True).decode('utf-8')
        content = "".join(trs.xpath(".//text()"))
        title = html.xpath("//meta[@name='ArticleTitle']/@content")[0]
        pubDate = html.xpath("//meta[@name='PubDate']/@content")[0][:10]
        front_url = re.findall("(.*\/)", resp_url)[0]
        if "dfnews" in resp_url:
            question_type = "地方快讯"
            htmlContent = htmlContent.replace("img src=\"./", f"img src=\"{front_url}")
        elif "hjywnews" in resp_url:
            htmlContent = htmlContent.replace("img src=\"./", f"img src=\"{front_url}")
            question_type = "环境要闻"
        else:
            question_type = ""
            # <img src="./
        item = {
            "url": resp_url,
            "uq_id": encode_md5(resp_url),
            "title": title,
            "pubDate": pubDate,
            'content': content,
            'htmlContent': htmlContent,
            'source': "中华人民共和国生态环境部",
            'category': "环保专栏",
            'question_type': question_type
        }
        yield item
if __name__ == '__main__':
    text = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, user-scalable=no, initial-scale=1, maximum-scale=1, minimum-scale=1">
<title>生态环境部一周要闻（4.10-4.16）_中华人民共和国生态环境部</title>
<meta name="SiteName" content="中华人民共和国生态环境部" />
<meta name="SiteDomain" content="www.mee.gov.cn" />
<meta name="SiteIDCode" content="bm17000009" />
<meta name="ColumnName" content="环境要闻" />
<meta name="ColumnType" content="环境要闻" />
<meta name="ArticleTitle" content="生态环境部一周要闻（4.10-4.16）"   />
<meta name="PubDate" content="2022-04-17 14:46:00"   />
<meta name="ContentSource" content="生态环境部"   />
<meta name="Keywords" content=""  />
<meta name="Author" content=""   />
<meta name="Url" content="http://www.mee.gov.cn/ywdt/hjywnews/202204/t20220417_974894.shtml">
<meta name="description" content="中华人民共和国生态环境部" />

<meta name="filetype" content="0"> 
<meta name="publishedtype" content="1"> 
<meta name="pagetype" content="1"> 
<meta name="catalogs" content="28943"> 
<meta name="contentid" content="974894"> 
<meta name="publishdate" content="2022-04-17"> 
<meta name="author" content="王丽平"> 
<meta name="source" content="生态环境部">
<script type="text/javascript" src="/images/jquery-1.12.4.min.js"></script>
<script type="text/javascript" src="/images/MEPC_mobile_v2020.js"></script>

<link class="stylefile" rel="stylesheet" type="text/css" href="/images/MEPC_base_pc_v2019.css" />
<link class="stylefile" rel="stylesheet" type="text/css" href="/images/MEPC_hzx_pc_v2019.css" />

<link class="mobileStyle" rel="stylesheet" type="text/css" href="/images/MEPC_base_mobile_v2019.css" />
<link class="mobileStyle" rel="stylesheet" type="text/css" href="/images/MEPC_hzx_mobile_v2019.css" />

<script class="stylefile" type="text/javascript" src="/images/respond.src_v2019.js"></script>
<script type="text/javascript" src="/images/jquery.SuperSlide.2.1.1.js"></script>
<script>
    var isPC = IsPC();//获取是否为PC端
    if (isPC) {//如果是PC端加载PC的js和css
        $(".mobileStyle").remove();
        dynamicLoadJs('/images/MEPC_base_pc_v2019.js')
        dynamicLoadJs('/images/MEPC_hzx_pc_v2019.js')
    } else {//如果是移动端加载移动端的js和css
        if(window.sessionStorage.getItem("isPc")=="true"){
            var oMeta = document.createElement('meta');
            oMeta.content = '';
            oMeta.name = 'viewport';
            document.getElementsByTagName('head')[0].appendChild(oMeta);
            $(".mobileStyle").remove();
            dynamicLoadJs('/images/MEPC_base_pc_v2019.js')
            dynamicLoadJs('/images/MEPC_hzx_pc_v2019.js')
            $(function(){
                $(".AppBtn").css('display','block');
                $(".style_App").css('display','block');
            })
        }else{
        $(".stylefile").remove();
        dynamicLoadJs('/images/MEPC_base_mobile_v2019.js')
        dynamicLoadJs('/images/MEPC_hzx_mobile_v2019.js')
        }
        $(function(){
            //切换pc移动
            $(".style_pc").click(function(){
                window.sessionStorage.setItem("isPc","true")
                window.location.reload()
            })

            $(".PcBtn").click(function(){
                window.sessionStorage.setItem("isPc","true")
                window.location.reload()
            })

            $(".style_App").click(function(){
                window.sessionStorage.setItem("isPc","false")
                window.location.reload()
            })

            $(".AppBtn").click(function(){
                window.sessionStorage.setItem("isPc","false")
                window.location.reload()
            })
        })
    }

    function dynamicLoadCss(url) {//加载css方法
        var head = document.getElementsByTagName('head')[0];
        var link = document.createElement('link');
        link.type = 'text/css';
        link.rel = 'stylesheet';
        link.href = url;
        head.appendChild(link);
    }

    function dynamicLoadJs(url, callback) {//加载js方法
        var head = document.getElementsByTagName('head')[0];
        var script = document.createElement('script');
        script.type = 'text/javascript';
        script.src = url;
        if (typeof(callback) == 'function') {
            script.onload = script.onreadystatechange = function () {
                if (!this.readyState || this.readyState === "loaded" || this.readyState === "complete") {
                    callback();
                    script.onload = script.onreadystatechange = null;
                }
            };
        }
        head.appendChild(script);
    }
</script>
</head>
<body>
<div class="header">
	<div class="headerC center">
		<!--<a class="backHome" href="">返回首页</a>-->
		<a class="AppBtn" target="_parent">手机版</a>
		<ul>
<!--
			<li class="languageBtn">
				<a class="languageBtnA">中文<img class="languageBtnImg" src="/images/MEPC_base_v2019_03.png" /></a>
				<div class="languageUl">
					<a href="/" target="_parent">中文</a>
					<a href="http://english.mee.gov.cn/" target="_blank">英文</a>
				</div>
			</li>
-->
			<li class="header_mailbox_li"><a href="https://email.mee.gov.cn/" target="_blank"><img src="/images/MEPC_base_topImg_v2019_08.png" />邮箱</a></li>
<li>
<script type="text/javascript">
    var ftzw = "tps://big5.mee.gov.cn/gate/big5/www.mee.gov.cn/";
    var jtzw = "tps://www.mee.gov.cn";
    var currUrl = document.URL;
    var flag = currUrl.indexOf("big5");
    if(flag<0)
        document.write('<a href="ht'+ftzw+'" target="_parent" title="繁体版">繁</a>');
    else
        document.write('<a href="ht'+jtzw+'" target="_parent" title="简体版">简</a>');
</script>
</li>
			<li><a href="http://english.mee.gov.cn/" target="_blank">EN</a></li>
			<li class="headerWx headerEwmBox">
				<a href="http://www.mee.gov.cn/home/wbwx/" target="_blank"><img src="/images/MEPC_base_topImg_v2019_03.png" /></a>
				<img class="header_ewmImg" src="/images/MEPC_base_top2_v2019_13.png" />
			</li>
			<li class="headerWb headerEwmBox">
				<a href="http://www.mee.gov.cn/home/wbwx/" target="_blank"><img src="/images/MEPC_base_topImg_v2019_05.png" /></a>
				<img class="header_ewmImg" src="/images/MEPC_base_toutiao_v2019_03.png" />
			</li>
			<li class="headerCLILast"><a href="javascript:void(0)"  id="cniil_wza" title="无障碍"><img src="/images/MEPC_base_topImg_v2019_06.png" /></a></li>
		</ul>
	</div>
</div>
<div class="header_logo">
	<div class="logoSearchBox center">
		<a class="logo" href="/" target="_parent"><img src="/images/MEPC_base_v2019_07.png" /></a>
		<div class="logoSearchR">
<script>
function checkForm(){
$("#searchword").val($.trim($("#searchword").val()));
	$("#searchword").val($("#searchword").val().replace(/请输入关键词/g, ''));
	
		$("#searchword").val($("#searchword").val().replace(/script/ig,''));
		$("#searchword").val($("#searchword").val().replace(/iframe/ig,''));
		$("#searchword").val($("#searchword").val().replace(/update/ig,''));
		$("#searchword").val($("#searchword").val().replace(/alert/ig,''));	

	if($("#searchword").val()==""){
		alert("请输入关键词");
		return false;
	}
}
</script>

			<div class="fl logoSearch">
<form action="/searchnew/" method="get" target="_blank" onsubmit="return checkForm();">
				<div class="logoSearchL">
					<div class="logoSearchLNei">
						<input type="submit" class="searchSubmit1" value="" />
						<input id="searchword" name="searchword" class="searchText1" type="text" placeholder="请输入您要搜索的内容" />
						<span class="clear" id="delete">×</span>
					</div>
					<input type="submit" class="searchSubmit2" value="搜索" />
					<a class="PcBtn">手机版</a>
				</div>
</form>
<script>
document.getElementById("searchword").addEventListener("keyup", function (){
	if (this.value.length > 0) {
		document.getElementById("delete").style.visibility = "visible";
		document.getElementById("delete").onclick = function () {
			document.getElementById("searchword").value = "";
			document.getElementById("delete").style.visibility = "hidden";
		}
	} else {
		document.getElementById("delete").style.visibility = "hidden";
	}
});
</script>
				<div class="searchWord">热门搜索：

<script>
var searchword = encodeURIComponent("环境影响评价");  
document.write("<a href='/searchnew?searchword="+searchword+"' target='_blank'>环境影响评价</a>")
</script>

<script>
var searchword = encodeURIComponent("空气质量");  
document.write("<a href='/searchnew?searchword="+searchword+"' target='_blank'>空气质量</a>")
</script>

				</div>
				<div class="fan_en">
<script type="text/javascript">
    var ftzw = "tp://big5.mee.gov.cn/gate/big5/www.mee.gov.cn/";
    var jtzw = "tps://www.mee.gov.cn";
    var currUrl = document.URL;
    var flag = currUrl.indexOf("big5");
    if(flag<0)
        document.write('<a href="ht'+ftzw+'" target="_parent" title="繁体版">繁</a>');
    else
        document.write('<a href="ht'+jtzw+'" target="_parent" title="简体版">简</a>');
</script>
<a href="http://english.mee.gov.cn/" target="_blank">EN</a>
				</div>

			</div>
			<a class="fl aqjBtn" href="http://nnsa.mee.gov.cn" target="_blank">
				<img class="aqjBtnPc" src="/images/MEPC_base_v2019_17.jpg" />
				<img class="aqjBtnYd" src="/images/MEPC_base_ydGjhaqj_v2019_03.png" />
				<span>点击进入</span>
			</a>
		</div>
	</div>
</div>
<div class="innerBaseBanner">
  <img class="innerBaseBannerImg" src="/images/MEPC_base_innerBanner_v2019_02.jpg" />
  <div class="center">
    <h2><script>
    var url="http://www.mee.gov.cn/ywdt/hjywnews/";
    if(url.indexOf(".cn/ywdt/")>0)
    {document.write("要闻动态");}
    if(url.indexOf(".cn/zjhb/")>0)
    {document.write("组织机构");}
    if(url.indexOf(".cn/zcwj/")>0)
    {document.write("政策文件");}
    if(url.indexOf(".cn/hjzl/")>0)
    {document.write("环境质量");}
    if(url.indexOf(".cn/ywgz/")>0)
    {document.write("业务工作");}
    if(url.indexOf(".cn/djgz/")>0)
    {document.write("机关党建");}
    if(url.indexOf(".cn/xxgk/")>0)
    {document.write("政府信息公开");}
    if(url.indexOf(".cn/zwfw/")>0)
    {document.write("政务服务");}
    if(url.indexOf(".cn/hdjl/")>0)
    {document.write("互动交流");}
    if(url.indexOf(".cn/ztzl/")>0)
    {document.write("专题专栏");}
    if(url.indexOf(".cn/home/ztbd/")>0)
    {document.write("专题专栏");}

    if(url.indexOf(".cn/hjzli/")>0)
    {document.write("污染防治");}
    if(url.indexOf(".cn/stbh/")>0)
    {document.write("生态保护");}
    if(url.indexOf(".cn/hyfs_12801/")>0)
    {document.write("核与辐射");}
    if(url.indexOf(".cn/gzfw_13107/")>0)
    {document.write("办事服务");}
</script></h2>
    <div class="crumbsNav">
      当前位置：<a href="../../../" target="_self" title="首页" class="CurrChnlCls">首页</a>&nbsp;>&nbsp;<a href="../../" target="_self" title="要闻动态" class="CurrChnlCls">要闻动态</a>&nbsp;>&nbsp;<a href="../" target="_self" title="环境要闻" class="CurrChnlCls">环境要闻</a>
    </div>
  </div>
</div>
<div class="innerBg">
  <div class="innerCenter">
    <div class="wzsmCenter">
      <div class="innerBzCenter">
        <div class="stbzXq">
          <div class="neiright_Box">
            <h2 class="neiright_Title">生态环境部一周要闻（4.10-4.16）</h2>
            <div class="neiright_Content">
              <div class="neiright_JPZ_GK">
                <p class="ydLyZzBox">
                  <span>2022-04-17</span>
                  <span>来源：生态环境部</span>
                </p>
                <span class="xqLyPc time">2022-04-17</span>
                <span class="xqLyPc">来源：生态环境部</span>
                <div class="neiright_JPZGK">分享到：
<div class="bshare-custom icon-medium" style="position: absolute;top: 24%;right: 0;">
    <a title="分享到微信" class="bshare-weixin"></a>
    <a title="分享到新浪微博" class="bshare-sinaminiblog"></a>
</div>
<script src="/images/buttonLite.js#style=1&amp;uuid=&amp;lang=zh" type="text/javascript" charset="utf-8"></script>
<script src="/images/bshareC0.js" type="text/javascript" charset="utf-8"></script>
</div>
                <span class="zzjgLdPrint"><a  href="javascript:void(1)" onclick="window.print()">[打印]</a></span>
                <span class="zzjgLdFont">字号：<a>[大]</a> <a class="active">[中]</a> <a >[小]</a></span>
              </div>
              <div class="neiright_JPZ_GK_CP">
                <div class=TRS_Editor><div class="Custom_UnionStyle">
<div>　　<b>1.生态环境部召开部常务会议</b></div>
<div>　　4月12日，生态环境部部长黄润秋主持召开部常务会议，审议并原则通过《六氟化铀运输容器》和《放射性固体废物近地表处置场辐射环境监测要求》2项国家标准。生态环境部党组书记孙金龙出席会议。&gt;&gt;&gt;<a href="http://www.mee.gov.cn/ywdt/hjywnews/202204/t20220412_974422.shtml">更多内容，点击阅读</a></div>
<div>　　<b>2.中国环境科学学会第九次全国会员代表大会召开</b></div>
<div>　　4月10日，中国环境科学学会第九次全国会员代表大会在京召开。生态环境部部长黄润秋，中国科学技术协会党组书记、分管日常工作副主席、书记处第一书记张玉卓出席会议并致辞。&gt;&gt;&gt;<a href="http://www.mee.gov.cn/ywdt/hjywnews/202204/t20220410_974151.shtml">更多内容，点击阅读</a></div>
<div>　　<b>3.第二轮第六批中央生态环境保护督察全面进入下沉工作阶段</b></div>
<div>　　经党中央、国务院批准，第二轮第六批5个中央生态环境保护督察组于2022年3月23日至25日陆续进驻河北、江苏、内蒙古、西藏、新疆五个省（区）和新疆生产建设兵团开展督察。截至目前，各督察组全面进入下沉工作阶段。&gt;&gt;&gt;<a href="http://www.mee.gov.cn/ywgz/zysthjbhdc/dcjl/202204/t20220411_974203.shtml">更多内容，点击阅读</a></div>
<div>　<b>　4.中央生态环境保护督察集中通报典型案例</b></div>
<div>　　第二轮第六批中央生态环境保护督察组深入一线、深入现场，查实了一批突出生态环境问题，核实了一批不作为、慢作为，不担当、不碰硬，甚至敷衍应对、弄虚作假等形式主义、官僚主义问题。为发挥警示作用，切实推动问题整改，中央生态环境保护督察集中公开对第二批5个典型案例进行通报。</div>
<div>　　▶ 典型案例丨<a href="http://www.mee.gov.cn/ywgz/zysthjbhdc/dcjl/202204/t20220414_974700.shtml">河北邯郸钢铁行业去产能存在乱象 产业结构调整落实不力</a></div>
<div>　　▶ 典型案例丨<a href="http://www.mee.gov.cn/ywgz/zysthjbhdc/dcjl/202204/t20220414_974699.shtml">江苏宿迁生态保护不到位 大运河宿迁段环境问题突出</a></div>
<div>　　▶ 典型案例丨<a href="http://www.mee.gov.cn/ywgz/zysthjbhdc/dcjl/202204/t20220414_974698.shtml">内蒙古鄂尔多斯棋盘井区域违法取水用水问题突出 生态环境影响严重</a></div>
<div>　　▶ 典型案例丨<a href="http://www.mee.gov.cn/ywgz/zysthjbhdc/dcjl/202204/t20220414_974697.shtml">西藏那曲色尼区砂石开采违法违规问题突出 严重破坏高寒草原生态环境</a></div>
<div>　　▶ 典型案例丨<a href="http://www.mee.gov.cn/ywgz/zysthjbhdc/dcjl/202204/t20220414_974695.shtml">新疆“乌昌石”区域大气污染防治推进不力 重污染天气多发</a></div>
<div>　<b>　5.大气环境监测卫星成功发射 减污降碳协同增效再添利器</b></div>
<div>　　4月16日2时16分，我国在太原卫星发射中心成功将大气环境监测卫星发射升空。该卫星将在国际上首次实现CO2的主动激光探测和大气细颗粒物的主被动结合探测，能够对气态污染物、云和气溶胶以及水生态、自然生态等环境要素进行大范围、全天时综合监测，同时可支撑开展气象、农业农村等行业的遥感监测应用工作。&gt;&gt;&gt;<a href="http://www.mee.gov.cn/ywdt/hjywnews/202204/t20220416_974889.shtml">更多内容，点击阅读</a></div>
<div>　<b>　6.生态环境部印发《关于加强排污许可执法监管的指导意见》</b></div>
<div>　　近日，经中央全面深化改革委员会审议通过，生态环境部印发了《关于加强排污许可执法监管的指导意见》。《指导意见》从总体要求、全面落实责任、严格执法监管、优化执法方式、强化支撑保障等五方面提出了22项具体要求，推动形成企业持证排污、政府依法监管、社会共同监督的生态环境执法监管新格局。&gt;&gt;&gt;<a href="http://www.mee.gov.cn/ywdt/xwfb/202204/t20220411_974303.shtml">更多内容，点击阅读</a></div>
<div>　　▶ <a href="http://www.mee.gov.cn/ywdt/zbft/202204/t20220411_974308.shtml">生态环境部执法局负责同志就出台《关于加强排污许可执法监管的指导意见》答记者问</a></div>
<div>　　▶<a href="http://www.mee.gov.cn/zcwj/zcjd/202204/t20220411_974307.shtml"> 一图读懂《关于加强排污许可执法监管的指导意见》</a></div>
<div>　　<b>7.生态环境部公布第二批生态环境执法典型案例（排污许可领域）</b></div>
<div>　　4月11日，生态环境部通报排污许可领域8个典型案例，这些案例涉及无证排污、以欺骗手段获取排污许可证、超许可排放浓度排放污染物、不按证排污、未按照许可证要求开展自行监测、未提交执行报告、未建立台账、未进行排污登记等违法行为，相关属地生态环境部门依据《排污许可管理条例》等相关法律法规对涉案违法单位和人员予以严惩。&gt;&gt;&gt;<a href="http://www.mee.gov.cn/ywdt/xwfb/202204/t20220411_974313.shtml">更多内容，点击阅读</a></div>
<div>　　<b>8.生态环境部固体废物与化学品司有关负责人就《尾矿污染环境防治管理办法》答记者问</b></div>
<div>　　生态环境部近日印发了《尾矿污染环境防治管理办法》（生态环境部令第26号），将于2022年7月1日起实施。生态环境部固体司有关负责人就《办法》出台的背景和主要内容等，回答了记者提问。&gt;&gt;&gt;<a href="http://www.mee.gov.cn/ywdt/zbft/202204/t20220413_974579.shtml">更多内容，点击阅读</a></div>
<div>　　▶<a href="http://www.mee.gov.cn/zcwj/zcjd/202204/t20220413_974578.shtml"> 一图读懂《尾矿污染环境防治管理办法》</a></div>
<div>　　<b>9.生态环境部发布4月下半月全国空气质量预报会商结果</b></div>
<div>　　2022年4月15日，中国环境监测总站联合中央气象台、国家大气污染防治攻关联合中心、东北、华南、西南、西北、长三角区域空气质量预测预报中心和北京市生态环境监测中心，开展4月下半月（16—30日）全国空气质量预报会商。4月下半月，全国大部扩散条件较好，空气质量以优良为主，局部地区可能出现轻度污染过程。其中，华东、东北局地可能出现中度污染过程，西北北部和新疆南疆地区受沙尘天气影响，可能出现重度及以上污染过程。&gt;&gt;&gt;<a href="http://www.mee.gov.cn/ywdt/xwfb/202204/t20220415_974878.shtml">更多内容，点击阅读</a></div>
<div>　　<b>10.《征程——@生态环境部在2021》出版发行</b></div>
<div>　　近日，由生态环境部组织编写，收录生态环境部政务新媒体2021年发布重点信息的图书——《征程——@生态环境部在2021》出版发行。本书以时间为轴，选取了生态环境部政务新媒体2021年发布的148篇重点信息，真实记录了我国生态环境保护工作的进展，客观反映了全国生态环保人深入打好污染防治攻坚战的奋斗历程，展现了社会各界携手推进生态文明建设的生动场景。本书是生态环境部连续第五年出版的政务新媒体实录图书。&gt;&gt;&gt;<a href="http://www.mee.gov.cn/ywdt/hjywnews/202204/t20220417_974896.shtml">更多内容，点击阅读</a></div>
</div></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!--信息挖掘 start-->
      <div class="stbzCBaseXq" id='recommend-rel' style="display: none">
        <div class="cjcs_com_tt">
          <h2>相关阅读推荐</h2>
        </div>
        <div class="cjcx_hdjl_liebadd">
          <ul>
            <!-- <li><span>2019-09-15</span><a href="" target="_blank">生态环境部通报2019年5月全国“12369”环保举报办理情况</a></li> -->
          </ul>
        </div>
      </div>

      <div class="stbzCBaseXq" id="recommend-int" style="display: none">
        <div class="cjcs_com_tt">
          <h2>您可能对以下文章感兴趣</h2>
        </div>
        <div class="cjcx_hdjl_liebadd">
          <ul>
            <!-- <li><span>2019-09-15</span><a href="" target="_blank">生态环境部通报2019年5月全国“12369”环保举报办理情况</a></li> -->
          </ul>
        </div>
      </div>
      <!--信息挖掘 end-->

      <div class="xqYdFx">

      </div>
      <div class="ydfxBox" >

      </div>
    </div>
  </div>
</div>
<div class="footer">
	<div class="footerTopUl">
		<div class="center">
			<ul class="active">
      
        <li><a href="http://www.fmprc.gov.cn/web/" target="_blank">外交部</a></li>
      
        <li><a href="http://www.mod.gov.cn/" target="_blank">国防部</a></li>
      
        <li><a href="http://www.ndrc.gov.cn" target="_blank">国家发展和改革委员会</a></li>
      
        <li><a href="http://www.moe.gov.cn" target="_blank">教育部</a></li>
      
        <li><a href="http://www.most.gov.cn/" target="_blank">科学技术部</a></li>
      
        <li><a href="http://www.miit.gov.cn/" target="_blank">工业和信息化部</a></li>
      
        <li><a href="http://www.seac.gov.cn/" target="_blank">国家民族事务委员会</a></li>
      
        <li><a href="http://www.mps.gov.cn/" target="_blank">公安部</a></li>
      
        <li><a href="http://www.mca.gov.cn/" target="_blank">民政部</a></li>
      
        <li><a href="http://www.moj.gov.cn/" target="_blank">司法部</a></li>
      
        <li><a href="http://www.mof.gov.cn/index.htm" target="_blank">财政部</a></li>
      
        <li><a href="http://www.mohrss.gov.cn/" target="_blank">人力资源社会保障部</a></li>
      
        <li><a href="http://www.mnr.gov.cn/" target="_blank">自然资源部</a></li>
      
        <li><a href="http://www.mee.gov.cn/" target="_blank">生态环境部</a></li>
      
        <li><a href="http://www.mohurd.gov.cn/" target="_blank">住房和城乡建设部</a></li>
      
        <li><a href="http://www.mot.gov.cn/" target="_blank">交通运输部</a></li>
      
        <li><a href="http://www.mwr.gov.cn/" target="_blank">水利部</a></li>
      
        <li><a href="http://www.moa.gov.cn/" target="_blank">农业农村部</a></li>
      
        <li><a href="http://www.mofcom.gov.cn/" target="_blank">商务部</a></li>
      
        <li><a href="https://www.mct.gov.cn/" target="_blank">文化和旅游部</a></li>
      
        <li><a href="http://www.nhc.gov.cn/" target="_blank">国家卫生健康委员会</a></li>
      
        <li><a href="http://www.mva.gov.cn/" target="_blank">退役军人事务部</a></li>
      
        <li><a href="https://www.mem.gov.cn/index.shtml" target="_blank">应急管理部</a></li>
      
        <li><a href="http://www.pbc.gov.cn/" target="_blank">人民银行</a></li>
      
        <li><a href="http://www.audit.gov.cn/" target="_blank">审计署</a></li>
      
        <li><a href="http://www.moe.gov.cn/jyb_sy/China_Language/" target="_blank">国家语委</a></li>
      
        <li><a href="http://www.cnsa.gov.cn" target="_blank">国家航天局</a></li>
      
        <li><a href="http://www.caea.gov.cn" target="_blank">国家原子能机构</a></li>
      
        <li><a href="http://nnsa.mee.gov.cn/" target="_blank">国家核安全局</a></li>
      
        <li><a href="http://www.sasac.gov.cn/" target="_blank">国务院国有资产监督管理委员会</a></li>
      
        <li><a href="http://www.customs.gov.cn/" target="_blank">海关总署</a></li>
      
        <li><a href="http://www.chinatax.gov.cn/" target="_blank">国家税务总局</a></li>
      
        <li><a href="http://www.samr.gov.cn/" target="_blank">国家市场监督管理总局</a></li>
      
        <li><a href="http://www.nrta.gov.cn/" target="_blank">国家广播电视总局</a></li>
      
        <li><a href="http://www.sport.gov.cn/" target="_blank">国家体育总局</a></li>
      
        <li><a href="http://www.stats.gov.cn/" target="_blank">国家统计局</a></li>
      
        <li><a href="http://www.cidca.gov.cn/" target="_blank">国家国际发展合作署</a></li>
      
        <li><a href="http://www.nhsa.gov.cn/" target="_blank">国家医疗保障局</a></li>
      
        <li><a href="http://www.counsellor.gov.cn/" target="_blank">国务院参事室</a></li>
      
        <li><a href="http://www.ggj.gov.cn/" target="_blank">国家机关事务管理局</a></li>
      
        <li><a href="http://www.cnca.gov.cn/" target="_blank">国家认证认可监督管理委员会</a></li>
      
        <li><a href="http://www.sac.gov.cn/" target="_blank">国家标准化管理委员会</a></li>
      
        <li><a href="http://www.ncac.gov.cn/" target="_blank">国家新闻出版署（国家版权局）</a></li>
      
        <li><a href="http://www.sara.gov.cn/" target="_blank">国家宗教事务局</a></li>
      
        <li><a href="http://www.gov.cn/guoqing/2005-12/26/content_2652073.htm" target="_blank">国务院研究室</a></li>
      
        <li><a href="http://www.gqb.gov.cn/" target="_blank">国务院侨务办公室</a></li>
      
        <li><a href="http://www.gwytb.gov.cn/" target="_blank">国务院台湾事务办公室</a></li>
      
        <li><a href="http://www.cac.gov.cn/" target="_blank">国家互联网信息办公室</a></li>
      
        <li><a href="http://www.scio.gov.cn/index.htm" target="_blank">国务院新闻办公室</a></li>
      
        <li><a href="http://www.xinhuanet.com/" target="_blank">新华通讯社</a></li>
      
        <li><a href="http://www.cas.cn/" target="_blank">中国科学院</a></li>
      
        <li><a href="http://www.cass.cn/" target="_blank">中国社会科学院</a></li>
      
        <li><a href="http://www.cae.cn/" target="_blank">中国工程院</a></li>
      
        <li><a href="http://www.drc.gov.cn/" target="_blank">国务院发展研究中心</a></li>
      
        <li><a href="http://www.cnr.cn/" target="_blank">中央广播电视总台</a></li>
      
        <li><a href="http://www.cma.gov.cn/" target="_blank">中国气象局</a></li>
      
        <li><a href="http://www.cbirc.gov.cn/" target="_blank">中国银行保险监督管理委员会</a></li>
      
        <li><a href="http://www.csrc.gov.cn/" target="_blank">中国证券监督管理委员会</a></li>
      
        <li><a href="http://www.ccps.gov.cn/" target="_blank">国家行政学院</a></li>
      
        <li><a href="http://www.lswz.gov.cn/" target="_blank">国家粮食和物资储备局</a></li>
      
        <li><a href="http://www.nea.gov.cn/" target="_blank">国家能源局</a></li>
      
        <li><a href="http://www.sastind.gov.cn/" target="_blank">国家国防科技工业局</a></li>
      
        <li><a href="http://www.tobacco.gov.cn/" target="_blank">国家烟草专卖局</a></li>
      
        <li><a href="http://www.forestry.gov.cn/" target="_blank">国家林业和草原局</a></li>
      
        <li><a href="http://www.nra.gov.cn/" target="_blank">国家铁路局</a></li>
      
        <li><a href="http://www.caac.gov.cn/" target="_blank">中国民用航空局</a></li>
      
        <li><a href="http://www.spb.gov.cn/" target="_blank">国家邮政局</a></li>
      
        <li><a href="http://www.ncha.gov.cn/" target="_blank">国家文物局</a></li>
      
        <li><a href="http://www.satcm.gov.cn/" target="_blank">国家中医药管理局</a></li>
      
        <li><a href="http://www.chinamine-safety.gov.cn/" target="_blank">国家矿山安全监察局</a></li>
      
        <li><a href="http://www.safe.gov.cn/" target="_blank">国家外汇管理局</a></li>
      
        <li><a href="http://www.cnipa.gov.cn/" target="_blank">国家知识产权局</a></li>
      
        <li><a href="https://www.nia.gov.cn/" target="_blank">国家移民管理局</a></li>
      
        <li><a href="http://www.forestry.gov.cn/" target="_blank">国家公园管理局</a></li>
      
        <li><a href="http://www.scs.gov.cn/" target="_blank">国家公务员局</a></li>
      
        <li><a href="http://www.saac.gov.cn/" target="_blank">国家档案局</a></li>
      
        <li><a href="http://www.gjbmj.gov.cn/" target="_blank">国家保密局</a></li>
      
        <li><a href="http://www.oscca.gov.cn/" target="_blank">国家密码管理局</a></li>
      
			</ul>
			<ul>
      
        <li><a href="http://hbdc.mee.gov.cn/" target="_blank">华北督察局</a></li>
      
        <li><a href="http://hddc.mee.gov.cn/" target="_blank">华东督察局</a></li>
      
        <li><a href="http://hndc.mee.gov.cn/" target="_blank">华南督察局</a></li>
      
        <li><a href="http://xbdc.mee.gov.cn/" target="_blank">西北督察局</a></li>
      
        <li><a href="http://xndc.mee.gov.cn/" target="_blank">西南督察局</a></li>
      
        <li><a href="http://dbdc.mee.gov.cn" target="_blank">东北督察局</a></li>
      
        <li><a href="http://nro.mee.gov.cn/" target="_blank">华北核与辐射安全监督站</a></li>
      
        <li><a href="http://ecro.mee.gov.cn/" target="_blank">华东核与辐射安全监督站</a></li>
      
        <li><a href="http://scro.mee.gov.cn/" target="_blank">华南核与辐射安全监督站</a></li>
      
        <li><a href="http://swnro.mee.gov.cn" target="_blank">西南核与辐射安全监督站</a></li>
      
        <li><a href="http://nero.mee.gov.cn/" target="_blank">东北核与辐射安全监督站</a></li>
      
        <li><a href="http://nwro.mee.gov.cn/" target="_blank">西北核与辐射安全监督站</a></li>
      
        <li><a href="http://cjjg.mee.gov.cn/" target="_blank">长江流域生态环境监督管理局</a></li>
      
        <li><a href="http://huanghejg.mee.gov.cn/" target="_blank">黄河流域生态环境监督管理局</a></li>
      
        <li><a href="http://huaihejg.mee.gov.cn/" target="_blank">淮河流域生态环境监督管理局</a></li>
      
        <li><a href="http://hhbhjg.mee.gov.cn/" target="_blank">海河流域北海海域生态环境监督管理局</a></li>
      
        <li><a href="http://zjnhjg.mee.gov.cn/" target="_blank">珠江流域南海海域生态环境监督管理局</a></li>
      
        <li><a href="http://sljg.mee.gov.cn/" target="_blank">松辽流域生态环境监督管理局</a></li>
      
        <li><a href="http://thdhjg.mee.gov.cn/" target="_blank">太湖流域东海海域生态环境监督管理局</a></li>
      
        <li><a href="http://www.craes.cn/" target="_blank">中国环境科学研究院</a></li>
      
        <li><a href="http://www.cnemc.cn/" target="_blank">中国环境监测总站</a></li>
      
        <li><a href="http://www.edcmep.org.cn/" target="_blank">中日友好环境保护中心（环境发展中心）</a></li>
      
        <li><a href="http://www.prcee.org" target="_blank">环境与经济政策研究中心</a></li>
      
        <li><a href="http://www.cenews.com.cn/" target="_blank">中国环境报社</a></li>
      
        <li><a href="http://www.cesp.com.cn/" target="_blank">中国环境出版集团有限公司</a></li>
      
        <li><a href="http://www.chinansc.cn/" target="_blank">核与辐射安全中心</a></li>
      
        <li><a href="http://www.fecomee.org.cn/" target="_blank">对外合作与交流中心</a></li>
      
        <li><a href="http://www.nies.org/" target="_blank">南京环境科学研究所</a></li>
      
        <li><a href="http://www.scies.org/" target="_blank">华南环境科学研究所</a></li>
      
        <li><a href="http://www.caep.org.cn/" target="_blank">环境规划院</a></li>
      
        <li><a href="http://www.china-eia.com/" target="_blank">环境工程评估中心</a></li>
      
        <li><a href="http://www.secmep.cn/" target="_blank">卫星环境应用中心</a></li>
      
        <li><a href="http://www.meescc.cn/" target="_blank">固体废物与化学品管理技术中心</a></li>
      
        <li><a href="http://www.chinaeic.net/" target="_blank">信息中心</a></li>
      
        <li><a href="http://www.ncsc.org.cn/" target="_blank">国家应对气候变化战略研究和国际合作中心</a></li>
      
        <li><a href="https://www.nmemc.org.cn/" target="_blank">国家海洋环境监测中心</a></li>
      
        <li><a href="http://www.tcare-mee.cn/" target="_blank">土壤与农业农村生态环境监管技术中心</a></li>
      
        <li><a href="http://www.chinaeol.net/" target="_blank">宣传教育中心</a></li>
      
        <li><a href="http://www.chinacses.org/" target="_blank">中国环境科学学会</a></li>
      
        <li><a href="http://www.cepf.org.cn/" target="_blank">中华环境保护基金会</a></li>
      
        <li><a href="http://www.cecrpa.org.cn/" target="_blank">中国生态文明研究与促进会</a></li>
      
        <li><a href="http://www.sepact.com/" target="_blank">北京会议与培训基地</a></li>
      
			</ul>
			<ul>
      
        <li><a href="http://sthjj.beijing.gov.cn/" target="_blank">北京市生态环境局</a></li>
      
        <li><a href="http://sthj.tj.gov.cn/" target="_blank">天津市生态环境局</a></li>
      
        <li><a href="http://hbepb.hebei.gov.cn/" target="_blank">河北省生态环境厅</a></li>
      
        <li><a href="http://sthjt.shanxi.gov.cn" target="_blank">山西省生态环境厅</a></li>
      
        <li><a href="http://sthjt.nmg.gov.cn/" target="_blank">内蒙古自治区生态环境厅</a></li>
      
        <li><a href="http://sthj.ln.gov.cn/" target="_blank">辽宁省生态环境厅</a></li>
      
        <li><a href="http://sthjt.jl.gov.cn/" target="_blank">吉林省生态环境厅</a></li>
      
        <li><a href="http://www.hljdep.gov.cn/" target="_blank">黑龙江省生态环境厅</a></li>
      
        <li><a href="http://sthj.sh.gov.cn/" target="_blank">上海市生态环境局</a></li>
      
        <li><a href="http://sthjt.jiangsu.gov.cn/" target="_blank">江苏省生态环境厅</a></li>
      
        <li><a href="http://sthjt.zj.gov.cn/" target="_blank">浙江省生态环境厅</a></li>
      
        <li><a href="http://sthjt.ah.gov.cn" target="_blank">安徽省生态环境厅</a></li>
      
        <li><a href="http://sthjt.fujian.gov.cn/" target="_blank">福建省生态环境厅</a></li>
      
        <li><a href="http://sthjt.jiangxi.gov.cn/" target="_blank">江西省生态环境厅</a></li>
      
        <li><a href="http://www.sdein.gov.cn/" target="_blank">山东省生态环境厅</a></li>
      
        <li><a href="http://sthjt.henan.gov.cn/" target="_blank">河南省生态环境厅</a></li>
      
        <li><a href="http://sthjt.hubei.gov.cn" target="_blank">湖北省生态环境厅</a></li>
      
        <li><a href="http://sthjt.hunan.gov.cn/" target="_blank">湖南省生态环境厅</a></li>
      
        <li><a href="http://gdee.gd.gov.cn/" target="_blank">广东省生态环境厅</a></li>
      
        <li><a href="http://sthjt.gxzf.gov.cn" target="_blank">广西壮族自治区生态环境厅</a></li>
      
        <li><a href="http://hnsthb.hainan.gov.cn/" target="_blank">海南省生态环境厅</a></li>
      
        <li><a href="http://sthjj.cq.gov.cn" target="_blank">重庆市生态环境局</a></li>
      
        <li><a href="http://sthjt.sc.gov.cn" target="_blank">四川省生态环境厅</a></li>
      
        <li><a href="http://sthj.guizhou.gov.cn/" target="_blank">贵州省生态环境厅</a></li>
      
        <li><a href="http://sthjt.yn.gov.cn/index.html" target="_blank">云南省生态环境厅</a></li>
      
        <li><a href="http://ee.xizang.gov.cn/" target="_blank">西藏自治区生态环境厅</a></li>
      
        <li><a href="http://sthjt.shaanxi.gov.cn" target="_blank">陕西省生态环境厅</a></li>
      
        <li><a href="http://sthj.gansu.gov.cn" target="_blank">甘肃省生态环境厅</a></li>
      
        <li><a href="http://sthjt.qinghai.gov.cn" target="_blank">青海省生态环境厅</a></li>
      
        <li><a href="https://sthjt.nx.gov.cn" target="_blank">宁夏回族自治区生态环境厅</a></li>
      
        <li><a href="http://sthjt.xinjiang.gov.cn/" target="_blank">新疆维吾尔自治区生态环境厅</a></li>
      
        <li><a href="http://sthjj.xjbt.gov.cn/?LMCL=HcSri1" target="_blank">新疆生产建设兵团生态环境局</a></li>
      
			</ul>
			<ul>
      
        <li><a href="http://dangxiao.mee.gov.cn/" target="_blank">生态环境部党校</a></li>
      
        <li><a href="https://news.cop15-china.com.cn" target="_blank">《生物多样性公约》COP15东道国网站</a></li>
      
        <li><a href="http://permit.mee.gov.cn/permitExt/defaults/default-index!getInformation.action" target="_blank">全国排污许可证管理信息平台</a></li>
      
        <li><a href="http://www.ozone.org.cn/" target="_blank">中国保护臭氧层行动</a></li>
      
        <li><a href="http://www.ccchina.org.cn" target="_blank">中国气候变化信息网</a></li>
      
			</ul>
		</div>
	</div>
	<div class="center">
		<div class="footer_web_linkBox">
			<!--<a class="footer_web_linkLA" href="http://www.gov.cn/" target="_blank">中国政府网</a>-->
			<ul class="footer_web_linkUl">
				<li class="footer_web_linkLi"><a class="footer_web_linkLiA" href="http://www.gov.cn/"  target="_blank">中国政府网</a></li>
				<li class="footer_web_linkLi footer_web_linkLi2">国务院部门<img src="/images/MEPC_base_footer_v2019_03.png" /></li>
				<li class="footer_web_linkLi footer_web_linkLi2">部系统门户网站群<img src="/images/MEPC_base_footer_v2019_03.png" /></li>
				<li class="footer_web_linkLi footer_web_linkLi2">地方生态环境部门<img src="/images/MEPC_base_footer_v2019_03.png" /></li>
				<li class="footer_web_linkLi footer_web_linkLi2">其他<img src="/images/MEPC_base_footer_v2019_03.png" /></li>
			</ul>
		</div>
		<div class="footer_web_linkBBox">
			<span>链接 ：</span>
			<ul>
				<li><a href="http://www.npc.gov.cn/"  target="_blank">全国人大</a></li>
				<li><a href="http://www.cppcc.gov.cn/"  target="_blank">全国政协</a></li>
				<li><a href="http://www.ccdi.gov.cn/"  target="_blank">国家监察委员会</a></li>
				<li><a href="http://www.court.gov.cn/"  target="_blank">最高人民法院</a></li>
				<li><a href="http://www.spp.gov.cn/"  target="_blank">最高人民检察院</a></li>
			</ul>
		</div>
		<div class="footer_BBox">
			<a href="http://bszs.conac.cn/sitename?method=show&id=1ACC49F8764D14F7E053012819ACEFF4" target="_blank"><img class="footer_BLImg" src="/images/MEPC_base_footer_v2019_14.png" /></a>
			<div class="footer_BZBox">
				<ul>
					<li><a href="/home/wzsm/" target="_blank">网站声明</a></li>
					<li><a href="/home/wzdt/" target="_blank">网站地图</a></li>
					<li><a href="/home/lxwm/" target="_blank">联系我们</a></li>
					<li><a href="http://www.govwza.cn/yxsm/?m=api&a=loadpc&sid=47933" target="_blank">无障碍客户端</a></li>
				</ul>
				<p>
					版权所有：中华人民共和国生态环境部<span>|</span>ICP备案编号: <a href="https://beian.miit.gov.cn " target="_blank" style="color:#000">京ICP备05009132号</a>
				</p>
				<p>
					网站标识码：bm17000009<span>|</span><a href="http://www.beian.gov.cn/portal/registerSystemInfo?recordcode=11040102700072" target="_blank" style="color:#000"><img src="/images/babs.png" /> 京公网安备 11040102700072号</a>
				</p>
			</div>
			<div class="footer_BRBox">
				<a><img src="/images/wuzhangai_app.gif" width="60" height="60" style="margin-left: 18px;" />无障碍APP安卓版</a>
				<a><img src="/images/icon_13.png" width="60" height="60" />手机版</a>
<a style="cursor: auto;"><img src="/images/2022slh.png" width="121" height="55" /></a>
<script id="_jiucuo_" sitecode='bm17000009' src='https://zfwzgl.www.gov.cn/exposure/jiucuo.js'></script>
<!--
				<a href="http://121.43.68.40/exposure/jiucuo.html?site_code=bm17000009" target="_blank"><img class="footer_BR_zfzc" src="/images/MEPC_base_footer_v2019_11.png" /></a>
-->
			</div>
			<a class="style_App" target="_parent">手机版</a>
		</div>
	</div>
</div>
<div class="footerMoble">
	<p>版权所有：中华人民共和国生态环境部 </p>
	<p>ICP备案编号: <a href="https://beian.miit.gov.cn " target="_blank" style="color:#000">京ICP备05009132号</a></p>
	<p>网站标识码：bm17000009</p>
	<p><a href="http://www.beian.gov.cn/portal/registerSystemInfo?recordcode=11040102700072" target="_blank" style="color:#000"><img src="/images/babs.png" /> 京公网安备 11040102700072号</a></p>
	<div class="style_pc">电脑版</div>
	<p class="footerMobleP">
		<a href="http://bszs.conac.cn/sitename?method=show&id=1ACC49F8764D14F7E053012819ACEFF4" target="_blank"><img class="footerMobleImg" src="/images/MEPC_base_footer_v2019_14.png" /></a>
		<a href="https://zfwzgl.www.gov.cn/exposure/jiucuo.html?site_code=bm17000009" target="_blank">
            <img class="footerMobleImg2" src="/images/MEPC_base_footer_v2019_11.png" />
        </a>
	</p>
	<a class="goTopBtn"></a>
</div>
<!--
<script type="text/javascript">document.write(unescape("%3Cspan id='_ideConac' %3E%3C/span%3E%3Cscript   src='https://dcs.conac.cn/js/33/000/0000/40672392/CA330000000406723920001.js' type='text/javascript'%3E%3C/script%3E"));</script>
-->
<script defer="" async="" type="text/javascript" src="//api.govwza.cn/cniil/assist.js?sid=47933&pos=left"></script>
<div style="display:none">
<script type="text/javascript">document.write(unescape("%3Cscript src='https://cl3.webterren.com/webdig.js?z=41' type='text/javascript'%3E%3C/script%3E"));</script>
<script type="text/javascript">wd_paramtracker("_wdxid=000000000000000000000000000000000000000000")</script>
</div>

<!-- 统计20200825-->
<script>
var _hmt = _hmt || [];
(function() {
  var hm = document.createElement("script");
  hm.src = "https://hm.baidu.com/hm.js?0f50400dd25408cef4f1afb556ccb34f";
  var s = document.getElementsByTagName("script")[0]; 
  s.parentNode.insertBefore(hm, s);
})();
</script>
<script>
$(function () {
    $("[href='####']").removeAttr("href");
});
</script>
<script type="text/javascript">
  var Recommender = (function() {
    this.docid = '974894';
    function getDocReltime(url) {
      var html = url.substr(url.lastIndexOf('/'));
      var year =html.substr(2,4);
      var month =html.substr(6,2);
      var day =html.substr(8,2);
      var docreltime = year + "-" + month + "-" + day
      return docreltime;
    }

    //检查当前用户访问的cookie,如果已经设置过cookie则跳过，没有则先设置cookie
    function checkAndGenerateCookie() {
      var viewsid = $.cookie("viewsid");
      if (!viewsid) {
        viewsid = guid();
        $.cookie('viewsid', viewsid, {expires: 3000,path:'/',domain:'.mee.gov.cn',secure:false,raw:false});
      }
      return viewsid;
    }

    //生成随机的cookieID
    function guid() {
      return 'xxxxxxxxxxxx4xxxyxxxxxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
        var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
      });
    }

    //加载相关阅读推荐
    this.loadSimilar = function() {
      $.ajax({
        url: "/api/v1/articles/similar",
        type: "GET",
        timeout: 20000,
        // data: {"id": 447656},
        data: {"id": docid},
        success: function(result) {
          if(result.code == 10000 && result.total != 0){
            $("#recommend-rel").find("ul").first().html('');
            for(var i = 0;i < 5 && i < result.total; i++){
              var str = '<li>' +
                  '<span>' + result.data[i].date + '</span>' +
                  '<a href="' + result.data[i].url + '" target="_blank">' + result.data[i].title + '</a>' +
                  '</li>';
              $("#recommend-rel").find("ul").first().append(str);
            }
            $("#recommend-rel").show();
          }
        }
      });
    }

    //加载感兴趣文章
    this.loadRecommend = function() {
      $.ajax({
        url: "/api/v2/articles/recommend",
        type: "POST",
        timeout: 20000,
        xhrFields: {withCredentials: true },
        data: {
          "viewsId": checkAndGenerateCookie()
        },
        success:function(result)   {
          if(result.code == 10000 && result.total != 0){
            $("#recommend-int").find("ul").first().html('');
            for(var i = 0; i < 5 && i < result.total; i++){
              var str = '<li>' +
                  '<span>' + getDocReltime(result.data[i].url) + '</span>' +
                  '<a href="' + result.data[i].url + '" target="_blank">' + result.data[i].title + '</a>' +
                  '</li>';
              $("#recommend-int").find("ul").first().append(str);
            }
            $("#recommend-int").show();
          }
        }
      });
    }

    //收集用户文章浏览记录
    this.collectionData = function() {
      $.ajax({
        url:"/api/v2/articles/views",
        type:"POST",
        timeout: 3000,
        xhrFields: {withCredentials: true},
        data: {
          "siteId": 2,
        "docId": 974894,
          "docTitle": "生态环境部一周要闻（4.10-4.16）",
          "url": "http://www.mee.gov.cn/ywdt/hjywnews/202204/t20220417_974894.shtml",
          "viewsId": checkAndGenerateCookie()
    },
      success: function (result) { }
    });
    }

    return {
      loadSimilar: loadSimilar,
      loadRecommend: loadRecommend,
      collectionData: collectionData
    }
  })();

  /**
   * jQuery 关于操作cookie的扩展
   **/
  jQuery.cookie = function(name, value, options) {
    if (typeof value != 'undefined') {
      options = options || {};
      if (value === null) {
        value = '';
        options = $.extend({}, options);
        options.expires = -1;
      }
      var expires = '';
      if (options.expires && (typeof options.expires == 'number' || options.expires.toUTCString)) {
        var date;
        if (typeof options.expires == 'number') {
          date = new Date();
          date.setTime(date.getTime() + (options.expires * 24 * 60 * 60 * 1000));
        } else {
          date = options.expires;
        }
        expires = '; expires=' + date.toUTCString();
      }
      var path = options.path ? '; path=' + (options.path) : '';
      var domain = options.domain ? '; domain=' + (options.domain) : '';
      var secure = options.secure ? '; secure' : '';
      document.cookie = [name, '=', encodeURIComponent(value), expires, path, domain, secure].join('');
    } else {
      var cookieValue = null;
      if (document.cookie && document.cookie != '') {
        var cookies = document.cookie.split(';');
        for (var i = 0; i < cookies.length; i++) {
          var cookie = jQuery.trim(cookies[i]);
          if (cookie.substring(0, name.length + 1) == (name + '=')) {
            cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
            break;
          }
        }
      }
      return cookieValue;
    }
  };


  $(document).ready(function(){
    Recommender.loadRecommend();
    Recommender.loadSimilar();
    Recommender.collectionData();
  })
</script>

<script type="text/javascript">
  /*查看点赞次数*/

  function checknum(){
    $.ajax({
      url:'/wcm-like/query.jsp?docid='+974894,
      dataType: 'JSON',
      type:'get',
      async:false,
      success:function(data){
        $(".dianzan .dz_cur").html(data+"人点赞");
      }, error:function(){
        //alert("访问错误");
      }
    });
  }
  checknum();
  $(".dianzan").click(function(){
    if($(this).hasClass("active"))
    {
      //alert("您已经点过赞了！");
    }else{
      $.ajax({
        url:'/wcm-like/like.jsp?docid='+974894,
        dataType: 'JSON',
        type:'get',
        async:false,
        success:function(data){
          //alert("点赞成功！");

        }, error:function(){
          //alert("访问错误");
        }
      });
      $(this).addClass('active');
      checknum();
    }

  });

     //适配table 宽度超出问题
      setTimeout(function(){  
           var winWidth=($(".stbzXq").width()); 
           $("table").each(function(i){ 
           var thisWidth = ($(this).width());
           if(winWidth<thisWidth ){
               $(this).css({'zoom':winWidth/thisWidth});
           }  
         }); 
     },200) 
     
     $(window).resize(function() {
          var winWidth=($(".stbzXq").width()); 
           $("table").each(function(i){ 
           var thisWidth = ($(this).width());
           if(winWidth<thisWidth ){
               $(this).css({'zoom':winWidth/thisWidth});
           }  
         }); 
     });
  //适配table 无边框问题
    setTimeout(function(){  
           $("table").each(function(i){ 
               var borderL = $(this).find("td").css("border-left");
               var borderR = $(this).find("td").css("border-right");
               var borderT = $(this).find("td").css("border-top");
               var borderB = $(this).find("td").css("border-bottom");
               if(borderL.indexOf("0px")>-1&&borderR.indexOf("0px")>-1&&borderT.indexOf("0px")>-1&&borderB.indexOf("0px")>-1){
                       $(this).css({"border-left":"1px solid #333","border-top":"1px solid #333"})
                       $(this).find("td").css({"border-right":"1px solid #333","border-bottom":"1px solid #333"})
                 }
           }); 
     },200) 
</script>
</body>
</html>"""
    # more_url = re.findall("<a href=\"(.*?)\">.*?更多内容，点击阅读</a>", text)
    more_url = re.findall("典型案例.*?<a href=\"(.*?)\">",text)
    print(more_url)
    url = "https://www.mee.gov.cn/ywgz/ydqhbh/qhbhlf/202210/t20221001_995515.shtml"
    print()