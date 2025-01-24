import scrapy

class BerlinServicesSpider(scrapy.Spider):
    name = "berlin_services"
    start_urls = [
        "https://service.berlin.de/dienstleistungen/"
    ]

    def parse(self, response):
        # Extract links to individual services
        service_links = response.xpath('//a[@href and contains(@href, "/dienstleistung/")]/@href').getall()

        for link in service_links:
            full_link = response.urljoin(link)
            yield scrapy.Request(full_link, callback=self.parse_service)

    def parse_service(self, response):
        # We are now on a service page
        title = response.xpath('//title/text()').get()
        h1_texts = response.xpath('//h1//text()').getall()
        h2_texts = response.xpath('//h2//text()').getall()

        all_text = response.xpath('//body//text()').getall()
        all_text_clean = ' '.join(t.strip() for t in all_text if t.strip())

        sub_links = response.xpath('//a[@href]/@href').getall()

        yield {
            'url': response.url,
            'title': title,
            'h1s': h1_texts,
            'h2s': h2_texts,
            'text': all_text_clean,
            'sub_links': sub_links
        }