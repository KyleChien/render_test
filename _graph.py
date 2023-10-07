from typing import List
import json 

class Node():
	def __init__(self, idx, title, tag, abstract, source, owner, status):
		self.id = idx
		self.title = title
		self.tag = tag
		self.abstract = abstract
		self.source = source
		self.owner = owner
		self.status = status

	def __repr__(self) -> str:
		return str(self.to_dict())

	@classmethod
	def from_dict(cls, dct):
		link = Link()
		for k, v in dct.items():
			setattr(link, k, v) 
		return link

	def to_dict(self):
		return {
			'id': self.id, 
			'title': self.title, 
			'tag': self.tag, 
			'abstract': self.abstract,
			'source': self.source, 
			'owner': self.owner,
			'status': self.status
		}

class Link():
	def __init__(self, source, target):
		self.source = source
		self.target = target

	def __repr__(self) -> str:
		return str(self.to_dict())

	@classmethod
	def from_dict(cls, dct):
		link = Link()
		for k, v in dct.items():
			setattr(link, k, v) 
		return link
	
	def to_dict(self):
		return {
			'source': self.source,
			'target': self.target
		}

class Graph():
	def __init__(self, nodes:List[Node], links:List[Link]):
		self.nodes = nodes
		self.links = links

	def __len__(self) -> int:
		return len(self.nodes)

	def __repr__(self) -> str:
		return str(self.to_dict())

	@classmethod
	def from_json(cls, json_path):
		with open(json_path, 'r') as f:
			data = json.load(f)
		return cls(data['nodes'], data['links'])

	def to_dict(self):
		return {
			'nodes': [node.to_dict() for node in self.nodes],
			'links': [link.to_dict() for link in self.links]
		}

	def to_json(self, json_path):
		dct = self.to_dict()
		with open(json_path, 'w') as f:
			json.dump(dct, f, indent=4)