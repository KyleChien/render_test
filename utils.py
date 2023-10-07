import torch
import json
from typing import List
from bardapi import BardCookies

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
		return cls(
			dct.get('id'),
			dct.get('title'),
			dct.get('tag'),
			dct.get('abstract'),
			dct.get('source'),
			dct.get('owner'),
			dct.get('status'),
		)

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
	def __init__(self, source, target, weight=1.0):
		self.source = source
		self.target = target
		self.weight = weight

	def __repr__(self) -> str:
		return str(self.to_dict())

	@classmethod
	def from_dict(cls, dct):
		return cls(
			dct.get('source'), 
			dct.get('target'), 
			dct.get('weight', 1.0)
		)
	
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
		nodes = [Node.from_dict(node) for node in data['nodes']]
		links = [Link.from_dict(link) for link in data['links']]
		return cls(nodes, links)

	def to_dict(self):
		return {
			'nodes': [node.to_dict() for node in self.nodes],
			'links': [link.to_dict() for link in self.links]
		}

	def to_json(self, json_path):
		dct = self.to_dict()
		with open(json_path, 'w') as f:
			json.dump(dct, f, indent=4)
	

def get_links_from_nodes(nodes:List[Node]) -> List[Link]:
	"""
	Build a list of link from tags of a list of node.
	"""

	return 

def autotag_with_Bloom(prompt, model, tokenizer, max_new_tokens=100, device='cpu'):
	inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
	with torch.cuda.amp.autocast():
		outputs = model.generate(inputs, max_new_tokens=max_new_tokens)
	return tokenizer.decode(outputs[0], skip_special_tokens=True)
