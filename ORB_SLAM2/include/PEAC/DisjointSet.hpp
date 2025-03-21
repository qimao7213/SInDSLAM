// Copyright (C) 2014,2023 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include <vector>

class DisjointSet
{
private:
	std::vector<int>	parent_;
	std::vector<int>	size_;

public:
	DisjointSet(const int n)
	{
		parent_.reserve(n);
		size_.reserve(n);

		for (int i=0; i<n; ++i)
		{
			parent_.push_back(i);
			size_.push_back(1);
		}
	}

	~DisjointSet() {}

	inline void remove(const int x) {
		if(parent_[x]!=x) {
			--size_[Find(x)];
			parent_[x]=x;
			size_[x]=1;
		}
	}

	inline int getSetSize(const int x) {
		return size_[Find(x)];
	}

	inline int Union(const int x, const int y)
	{
		const int xRoot = Find(x);
		const int yRoot = Find(y);

		if (xRoot == yRoot)
			return xRoot;

		const int xRootSize = size_[xRoot];
		const int yRootSize = size_[yRoot];

		if (xRootSize < yRootSize) {
			parent_[xRoot] = yRoot;
			size_[yRoot]+=size_[xRoot];
			return yRoot;
		} else {
			parent_[yRoot] = xRoot;
			size_[xRoot]+=size_[yRoot];
			return xRoot;
		}
	}

	inline int Find(const int x)
	{
		if (parent_[x] != x)
			parent_[x] = Find(parent_[x]);

		return parent_[x];
	}

private:
	DisjointSet();
	DisjointSet(const DisjointSet& rhs);
	const DisjointSet& operator=(const DisjointSet& rhs);
};
