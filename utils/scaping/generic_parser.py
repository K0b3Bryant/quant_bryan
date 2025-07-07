from .base_parser import BaseParser

class GenericListParser(BaseParser):
    """A powerful parser for pages that list items in a structured way."""
    def parse(self, soup) -> list:
        p_config = self.config['parser_config']
        items = soup.select(p_config['item_selector'])
        results = []

        for item in items:
            data = {}
            for field_name, field_config in p_config['fields'].items():
                selector = field_config['selector']
                attribute = field_config.get('attribute')
                field_type = field_config.get('type', 'string')

                if field_type == 'list':
                    # Extract a list of strings
                    elements = item.select(selector)
                    data[field_name] = [el.get_text(strip=True) for el in elements]
                else:
                    # Extract a single string or attribute
                    element = item.select_one(selector)
                    if element:
                        if attribute:
                            data[field_name] = element.get(attribute, '').strip()
                        else:
                            data[field_name] = element.get_text(strip=True)
                    else:
                        data[field_name] = None
            
            results.append(data)
        
        return results

