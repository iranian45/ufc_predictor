import scrapy
import pandas as pd
import string
# from scrapy.crawler import CrawlerProcess

class FightersURLSpider(scrapy.Spider):
    name = "fighter_urls"

    def start_requests(self):
        links = []
        alphabet = string.ascii_lowercase

        for letter in alphabet:
            link = 'http://www.ufcstats.com/statistics/fighters?char=' + letter + '&page=all'
            links.append(link)

        for url in links:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response, **kwargs):

        for fighter in response.xpath('//*[@class="b-statistics__table"]//tbody/tr')[1:]:
            yield {
                'fighter_url': fighter.xpath('.//td[1]/a//@href').extract_first(),
            }


urls_csv = pd.read_csv('G:\\Python\\UFC_Predictor\\UFC_Stats_Scrapper\\UFC_Stats_Scrapper\\spiders\\CSV'
                       '\\fighter_urls.csv')
urls_list = urls_csv.fighter_url.to_list()


class FightersDetailsSpider(scrapy.Spider):
    name = "fighter_details"

    def start_requests(self):
        links = []

        for link in urls_list:
            links.append(link)

        for url in links:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response, **kwargs):

        for fighter_details in response.xpath('/html/body/section/div/div'):
            yield {
                'Fighter_URL': response.request.url,
                'Height': fighter_details.xpath(
                    '/html/body/section/div/div/div[1]/ul/li[1]/text()[2]').extract_first().strip(),
                'Weight': fighter_details.xpath(
                    '/html/body/section/div/div/div[1]/ul/li[2]/text()[2]').extract_first().strip().split()[0],
                'Reach': fighter_details.xpath(
                    '/html/body/section/div/div/div[1]/ul/li[3]/text()[2]').extract_first().strip(),
                'Stance': fighter_details.xpath(
                    '/html/body/section/div/div/div[1]/ul/li[4]/text()[2]').extract_first().strip(),
                'DoB': fighter_details.xpath(
                    '/html/body/section/div/div/div[1]/ul/li[5]/text()[2]').extract_first().strip(),
            }


class FightUrlsSpider(scrapy.Spider):
    name = "fight_info"

    def start_requests(self):
        links = []

        for link in urls_list:
            links.append(link)

        for url in links:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response, **kwargs):

        for fight_url in response.xpath('//*[@class="b-fight-details__table-body"]//tr')[1:]:
            yield {
                'Fight_URL':
                    None if fight_url.xpath('.//td[1]/p/a/i/i/text()').extract_first().strip() == "next"
                    else fight_url.xpath('.//@href').extract_first(),
                'Fight_Date':
                    fight_url.xpath('.//td[6]/p[2]').extract_first().split("\n")[-2].strip() if fight_url.xpath(
                        './/td[1]/p/a/i/i/text()').extract_first().strip() == "next"
                    else fight_url.xpath('.//td[7]/p[2]').extract_first().split("\n")[-2].strip(),
            }


fight_info_csv = pd.read_csv('G:\\Python\\UFC_Predictor\\UFC_Stats_Scrapper\\UFC_Stats_Scrapper\\spiders\\CSV'
                             '\\fight_info.csv')
fight_urls_list = fight_info_csv.Fight_URL.to_list()

# remove nan
fight_urls_list = [fight_urls_list for fight_urls_list in fight_urls_list if str(fight_urls_list) != 'nan']


class FightDetailsSpider(scrapy.Spider):
    name = "fight_details"

    def start_requests(self):
        links = []

        for link in fight_urls_list:
            links.append(link)

        for url in links:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response, **kwargs):

        for fight_details in response.xpath('/html/body/section/div'):
            yield {
                'Fight_URL': response.request.url,
                'Event_Name': fight_details.xpath('//h2/a/text()').extract_first().strip(),
                'Event_URL': fight_details.xpath('//h2/a/@href').extract_first(),
                'Fight_Title': fight_details.xpath('//div/div[2]/div[1]/i').extract_first().split('\n')[-2].strip(),
                'Method': fight_details.xpath('//div/div[2]/div[2]/p[1]/i[1]/i[2]/text()').extract_first().strip(),
                'Round_Finished': fight_details.xpath('//div/div[2]/div[2]/p[1]/i[2]').extract_first().split()[-2],
                'Referee': fight_details.xpath('//div/div[2]/div[2]/p[1]/i[5]/span/text()').extract_first().strip(),
                'Stop_Time': fight_details.xpath('//div/div[2]/div[2]/p[1]/i[3]').extract_first().split()[-2],
                'F1_URL': fight_details.xpath('//div/div[1]/div[1]/div/h3/a/@href').extract_first(),
                'F1_First': fight_details.xpath('//div/div[1]/div[1]/div/h3/a/text()').extract_first().split()[
                    0].strip(),
                'F1_Last': fight_details.xpath('//div/div[1]/div[1]/div/h3/a/text()').extract_first().split()[
                    1].strip(),
                'F1_Status': fight_details.xpath('//div/div[1]/div[1]/i/text()').extract_first().strip(),
                'F1_KD': fight_details.xpath(
                    '//div/section[2]/table/tbody/tr/td[2]/p[1]/text()').extract_first().strip(),
                'F1_Total_Str_Landed':
                    fight_details.xpath('//div/section[2]/table/tbody/tr/td[5]/p[1]/text()').extract_first().split()[
                        0].strip(),
                'F1_Total_Str_Attempted':
                    fight_details.xpath('//div/section[2]/table/tbody/tr/td[5]/p[1]/text()').extract_first().split()[
                        2].strip(),
                'F1_Sig_Str_Landed':
                    fight_details.xpath('//div/section[2]/table/tbody/tr/td[3]/p[1]/text()').extract_first().split()[
                        0].strip(),
                'F1_Sig_Str_Attempted':
                    fight_details.xpath('//div/section[2]/table/tbody/tr/td[3]/p[1]/text()').extract_first().split()[
                        2].strip(),
                'F1_Sig_Str_Head_Landed':
                    fight_details.xpath('//div/table/tbody/tr/td[4]/p[1]/text()').extract_first().split()[0].strip(),
                'F1_Sig_Str_Head_Attempted':
                    fight_details.xpath('//div/table/tbody/tr/td[4]/p[1]/text()').extract_first().split()[2].strip(),
                'F1_Sig_Str_Body_Landed':
                    fight_details.xpath('//div/table/tbody/tr/td[5]/p[1]/text()').extract_first().split()[0].strip(),
                'F1_Sig_Str_Body_Attempted':
                    fight_details.xpath('//div/table/tbody/tr/td[5]/p[1]/text()').extract_first().split()[2].strip(),
                'F1_Sig_Str_Leg_Landed':
                    fight_details.xpath('//div/table/tbody/tr/td[6]/p[1]/text()').extract_first().split()[0].strip(),
                'F1_Sig_Str_Leg_Attempted':
                    fight_details.xpath('//div/table/tbody/tr/td[6]/p[1]/text()').extract_first().split()[2].strip(),
                'F1_Sig_Str_Distance_Landed':
                    fight_details.xpath('//div/table/tbody/tr/td[7]/p[1]/text()').extract_first().split()[0].strip(),
                'F1_Sig_Str_Distance_Attempted':
                    fight_details.xpath('//div/table/tbody/tr/td[7]/p[1]/text()').extract_first().split()[2].strip(),
                'F1_Sig_Str_Clinch_Landed':
                    fight_details.xpath('//div/table/tbody/tr/td[8]/p[1]/text()').extract_first().split()[0].strip(),
                'F1_Sig_Str_Clinch_Attempted':
                    fight_details.xpath('//div/table/tbody/tr/td[8]/p[1]/text()').extract_first().split()[2].strip(),
                'F1_Sig_Str_Ground_Landed':
                    fight_details.xpath('//div/table/tbody/tr/td[9]/p[1]/text()').extract_first().split()[0].strip(),
                'F1_Sig_Str_Ground_Attempted':
                    fight_details.xpath('//div/table/tbody/tr/td[9]/p[1]/text()').extract_first().split()[2].strip(),
                'F1_TD_Landed':
                    fight_details.xpath('//div/section[2]/table/tbody/tr/td[6]/p[1]/text()').extract_first().split()[
                        0].strip(),
                'F1_TD_Attempted':
                    fight_details.xpath('//div/section[2]/table/tbody/tr/td[6]/p[1]/text()').extract_first().split()[
                        2].strip(),
                'F1_Sub_Attempted': fight_details.xpath(
                    '//div/section[2]/table/tbody/tr/td[8]/p[1]/text()').extract_first().strip(),
                'F1_Rev': fight_details.xpath(
                    '//div/section[2]/table/tbody/tr/td[9]/p[1]/text()').extract_first().strip(),
                'F1_CTRL_TIME': fight_details.xpath(
                    '//div/section[2]/table/tbody/tr/td[10]/p[1]/text()').extract_first().strip(),
                'F2_URL': fight_details.xpath('//div/div[1]/div[2]/div/h3/a/@href').extract_first(),
                'F2_First': fight_details.xpath('//div/div[1]/div[2]/div/h3/a/text()').extract_first().split()[
                    0].strip(),
                'F2_Last': fight_details.xpath('//div/div[1]/div[2]/div/h3/a/text()').extract_first().split()[
                    1].strip(),
                'F2_Status': fight_details.xpath('//div/div[1]/div[2]/i/text()').extract_first().strip(),
                'F2_KD': fight_details.xpath(
                    '//div/section[2]/table/tbody/tr/td[2]/p[2]/text()').extract_first().strip(),
                'F2_Total_Str_Landed':
                    fight_details.xpath('//div/section[2]/table/tbody/tr/td[5]/p[2]/text()').extract_first().split()[
                        0].strip(),
                'F2_Total_Str_Attempted':
                    fight_details.xpath('//div/section[2]/table/tbody/tr/td[5]/p[2]/text()').extract_first().split()[
                        2].strip(),
                'F2_Sig_Str_Landed':
                    fight_details.xpath('//div/section[2]/table/tbody/tr/td[3]/p[2]/text()').extract_first().split()[
                        0].strip(),
                'F2_Sig_Str_Attempted':
                    fight_details.xpath('//div/section[2]/table/tbody/tr/td[3]/p[2]/text()').extract_first().split()[
                        2].strip(),
                'F2_TD_Landed':
                    fight_details.xpath('//div/section[2]/table/tbody/tr/td[6]/p[2]/text()').extract_first().split()[
                        0].strip(),
                'F2_TD_Attempted':
                    fight_details.xpath('//div/section[2]/table/tbody/tr/td[6]/p[2]/text()').extract_first().split()[
                        2].strip(),
                'F2_Sub_Attempted': fight_details.xpath(
                    '//div/section[2]/table/tbody/tr/td[8]/p[2]/text()').extract_first().strip(),
                'F2_Rev': fight_details.xpath(
                    '//div/section[2]/table/tbody/tr/td[9]/p[2]/text()').extract_first().strip(),
                'F2_CTRL_TIME': fight_details.xpath(
                    '//div/section[2]/table/tbody/tr/td[10]/p[2]/text()').extract_first().strip(),
                'F2_Sig_Str_Head_Landed':
                    fight_details.xpath('//div/table/tbody/tr/td[4]/p[2]/text()').extract_first().split()[0].strip(),
                'F2_Sig_Str_Head_Attempted':
                    fight_details.xpath('//div/table/tbody/tr/td[4]/p[2]/text()').extract_first().split()[2].strip(),
                'F2_Sig_Str_Body_Landed':
                    fight_details.xpath('//div/table/tbody/tr/td[5]/p[2]/text()').extract_first().split()[0].strip(),
                'F2_Sig_Str_Body_Attempted':
                    fight_details.xpath('//div/table/tbody/tr/td[5]/p[2]/text()').extract_first().split()[2].strip(),
                'F2_Sig_Str_Leg_Landed':
                    fight_details.xpath('//div/table/tbody/tr/td[6]/p[2]/text()').extract_first().split()[0].strip(),
                'F2_Sig_Str_Leg_Attempted':
                    fight_details.xpath('//div/table/tbody/tr/td[6]/p[2]/text()').extract_first().split()[2].strip(),
                'F2_Sig_Str_Distance_Landed':
                    fight_details.xpath('//div/table/tbody/tr/td[7]/p[2]/text()').extract_first().split()[0].strip(),
                'F2_Sig_Str_Distance_Attempted':
                    fight_details.xpath('//div/table/tbody/tr/td[7]/p[2]/text()').extract_first().split()[2].strip(),
                'F2_Sig_Str_Clinch_Landed':
                    fight_details.xpath('//div/table/tbody/tr/td[8]/p[2]/text()').extract_first().split()[0].strip(),
                'F2_Sig_Str_Clinch_Attempted':
                    fight_details.xpath('//div/table/tbody/tr/td[8]/p[2]/text()').extract_first().split()[2].strip(),
                'F2_Sig_Str_Ground_Landed':
                    fight_details.xpath('//div/table/tbody/tr/td[9]/p[2]/text()').extract_first().split()[0].strip(),
                'F2_Sig_Str_Ground_Attempted':
                    fight_details.xpath('//div/table/tbody/tr/td[9]/p[2]/text()').extract_first().split()[2].strip(),
            }


class NextFightUrlsSpider(scrapy.Spider):
    name = "next_fight_info"

    def start_requests(self):
        links = []

        for link in urls_list:
            links.append(link)

        for url in links:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response, **kwargs):

        for fight_url in response.xpath('//*[@class="b-fight-details__table-body"]//tr')[1:]:
            yield {
                'F1_URL': response.request.url,
                'F1_First': fight_url.xpath('.//td[2]/p[1]/a/text()').extract_first().split()[0].strip(),
                'F1_Last': fight_url.xpath('.//td[2]/p[1]/a/text()').extract_first().split()[1].strip(),
                'F2_URL': fight_url.xpath('.//td[2]/p[2]/a/@href').extract_first(),
                'F2_First': fight_url.xpath('.//td[2]/p[2]/a/text()').extract_first().split()[0].strip(),
                'F2_Last': fight_url.xpath('.//td[2]/p[2]/a/text()').extract_first().split()[1].strip(),
                'Fight_URL':
                    None if fight_url.xpath('.//td[1]/p/a/i/i/text()').extract_first().strip() != "next"
                    else fight_url.xpath('.//td[4]/p/a/@href').extract_first(),
                'Fight_Date':
                    fight_url.xpath('.//td[6]/p[2]').extract_first().split("\n")[-2].strip() if fight_url.xpath(
                        './/td[1]/p/a/i/i/text()').extract_first().strip() == "next"
                    else fight_url.xpath('.//td[7]/p[2]').extract_first().split("\n")[-2].strip(),
            }

class NextFightSchedule(scrapy.Spider):
    name = "next_fight_schedule"

    def start_requests(self):
        start_urls = [
            'http://www.ufcstats.com/event-details/277ffed20cf07aea'
        ]
        for url in start_urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response, **kwargs):

        for fighter in response.xpath('//*[@class="b-fight-details__table b-fight-details__table_style_margin-top b-fight-details__table_type_event-details js-fight-table"]//tbody/tr'):
            yield {
                'F1_URL': fighter.xpath('.//td[2]/p[1]/a/@href').extract_first(),
                'F1_First': fighter.xpath('.//td[2]/p[1]/a/text()').extract_first().split()[0].strip(),
                'F1_Last': fighter.xpath('.//td[2]/p[1]/a//text()').extract_first().split()[1].strip(),
                'F2_URL': fighter.xpath('./td[2]/p[2]/a//@href').extract_first(),
                'F2_First': fighter.xpath('./td[2]/p[2]/a//text()').extract_first().split()[0].strip(),
                'F2_Last': fighter.xpath('./td[2]/p[2]/a//text()').extract_first().split()[1].strip(),
            }


# process = CrawlerProcess({
#     'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)'
# })

# process.crawl(FightersURLSpider)
# process.crawl(FightersDetailsSpider)
# process.crawl(FightUrlsSpider)
# process.crawl(FightDetailsSpider)
# process.crawl(NextFightSchedule)
# process.start()
